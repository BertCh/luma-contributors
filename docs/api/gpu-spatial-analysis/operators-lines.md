---
title: "GPU Spatial Analysis: Lines & trajectories"
sidebar_label: "Lines & trajectories"
description: "Lines & trajectories contributors in @luma-contributors/analysis/gpu-spatial-analysis."
---


# Lines & trajectories

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.


### `GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`, and `GPULineChunk`

`GPULineSmooth`: Optional source-row and source-chainage metadata follows the emitted vertices.
Metadata rows are at most 1,073,741,823 and position addresses at most 2,147,483,647.

**`GPULineSegmentize`** — Densifies each polyline so no segment exceeds a maximum length, writing new vertices and source-line indices.

::lens{scenario=gpu-spatial-analysis.line-segmentize rung=result caption="Grey: 223 Wicker Park streets · blue: densified to segments of at most 25 m"}

**`GPUGreatCircleArcs`** — Tessellates a great-circle arc between each origin and destination pair into equal-angle vertices, writing one line per pair.

::lens{scenario=gpu-spatial-analysis.great-circle-arcs rung=result caption="Grey: 40 destinations · blue: great-circle arcs from ORD, 32 segments each"}

**`GPULineSmooth`** — Smooths each polyline with Chaikin corner cutting, writing the refined vertices and source-line indices.

::lens{scenario=gpu-spatial-analysis.line-smooth rung=result caption="Grey: 223 Wicker Park streets · blue: two Chaikin rounds with ratio 0.25"}

**`GPULineChunk`** — Splits each polyline into pieces of a fixed length, writing the pieces with their source-line indices.

::lens{scenario=gpu-spatial-analysis.line-chunk rung=result caption="Grey: 223 Wicker Park streets · blue: 100 m chunks"}

Geometry-producing line contributors. Each writes a `GPULineGeometryOutput`: flat `positions`, clamped
`lineOffsets` (deck.gl `PathLayer` `startIndices`), `status` (`count`, `overflow`, and optional `requiredCount`,
`approximation`, `vertexCount` and `requiredVertexCount`), optional `sourceLineIndices` (output line to input line), `sourceRows` (output vertex to input
segment row) and `measures` (distance along the source line). As in the generated-geometry layout
(contributor contract rule 8), `count` and `requiredCount` count output lines and `vertexCount` and
`requiredVertexCount` count vertices; `vertexCount` is the indirect draw count of `positions`.
**Breaking (2026-11-10):** `count` and `requiredCount` counted vertices before, and `status.lineCount`
is removed (it is `count`). A line truncated by the vertex capacity still counts as written. The vertex capacity is
`positions.length`; offsets are clamped to it, so a truncated result is still a valid layout.
Inputs are packed `positions` sorted by line with `lineOffsets` (`lineCount + 1` rows).
`GPULineSegmentize`, `GPULineChunk` and `GPUGreatCircleArcs` declare `measure: 'distance'`;
Lambert equal-area local frames throw (use an azimuthal-equidistant frame for planar lines).

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPULineSegmentize, getGPULineSegmentizeParameterValues, GPUGreatCircleArcs, getGPUGreatCircleArcsParameterValues, GPULineSmooth, GPULineChunk, getGPULineChunkParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample0(graph: GPUCommandGraph, inputs: {
  positions: NonNullable<ConstructorParameters<typeof GPULineSegmentize>[0]['positions']>;
  lineOffsets: NonNullable<ConstructorParameters<typeof GPULineSegmentize>[0]['lineOffsets']>;
  segmentizeParameters: GPUParameterBuffer<'float32'>;
  densified: NonNullable<NonNullable<ConstructorParameters<typeof GPULineSegmentize>[0]['output']>['positions']>;
  densifiedOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPULineSegmentize>[0]['output']>['lineOffsets']>;
  count: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULineSegmentize>[0]['output']>['status']>['count']>;
  overflow: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULineSegmentize>[0]['output']>['status']>['overflow']>;
  vertexCount: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULineSegmentize>[0]['output']>['status']>['vertexCount']>;
  measures: NonNullable<NonNullable<ConstructorParameters<typeof GPULineSegmentize>[0]['output']>['measures']>;
  sources: NonNullable<ConstructorParameters<typeof GPUGreatCircleArcs>[0]['sources']>;
  targets: NonNullable<ConstructorParameters<typeof GPUGreatCircleArcs>[0]['targets']>;
  arcParameters: GPUParameterBuffer<'float32'>;
  arcPositions: NonNullable<NonNullable<ConstructorParameters<typeof GPUGreatCircleArcs>[0]['output']>['positions']>;
  arcOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPUGreatCircleArcs>[0]['output']>['lineOffsets']>;
  smoothParameters: GPUParameterBuffer<'float32'>;
  output: NonNullable<ConstructorParameters<typeof GPULineSmooth>[0]['output']>;
  chunkParameters: GPUParameterBuffer<'float32'>;
  pieces: NonNullable<NonNullable<ConstructorParameters<typeof GPULineChunk>[0]['output']>['positions']>;
  pieceOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPULineChunk>[0]['output']>['lineOffsets']>;
  sourceLineIndices: NonNullable<NonNullable<ConstructorParameters<typeof GPULineChunk>[0]['output']>['sourceLineIndices']>;
}) {
  const {positions, lineOffsets, segmentizeParameters, densified, densifiedOffsets, count, overflow, vertexCount, measures, sources, targets, arcParameters, arcPositions, arcOffsets, smoothParameters, output, chunkParameters, pieces, pieceOffsets, sourceLineIndices} = inputs;
  graph.add(new GPULineSegmentize({
    positions, lineOffsets,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}, // omit for planar
    parameters: segmentizeParameters.importToGraph(graph),
    output: {positions: densified, lineOffsets: densifiedOffsets, status: {count, overflow, vertexCount}, measures}
  }));
  segmentizeParameters.write(getGPULineSegmentizeParameterValues({maximumSegmentLength: 50000}));

  graph.add(new GPUGreatCircleArcs({
    sources, targets, // longitude/latitude pairs
    parameters: arcParameters.importToGraph(graph),
    output: {positions: arcPositions, lineOffsets: arcOffsets, status: {count, overflow, vertexCount}}
  }));
  arcParameters.write(getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 0, minimumSegments: 32}));

  graph.add(new GPULineSmooth({positions, lineOffsets, iterationCount: 3, closed: false,
    parameters: smoothParameters.importToGraph(graph), output}));

  graph.add(new GPULineChunk({positions, lineOffsets, mode: 'chunk', // or 'substring'
    parameters: chunkParameters.importToGraph(graph),
    output: {positions: pieces, lineOffsets: pieceOffsets, status: {count, overflow, vertexCount}, sourceLineIndices}}));
  chunkParameters.write(getGPULineChunkParameterValues({chunkLength: 1000}));
}
```

- `GPULineSegmentize` (PostGIS `ST_Segmentize`, densify): every segment of length `L` becomes
  `clamp(ceil(L / maximumSegmentLength), 1, maximumPieceCount)` equal pieces; input vertices
  are kept. With compile-time `closedLines: true` (default false), each line is a ring that closes
  implicitly: the closing edge is densified and no vertex is repeated. A last row that repeats
  the first row bit-exactly is dropped; other repeated input vertices are preserved. Output keeps
  one line per input line, including empty and single-vertex lines, following the implicit polygon
  ring convention used by `GPURectangleClip`. The spherical closing edge follows the short way
  across the antimeridian using the last row's longitude shift.
  With a great-circle `spatialContext`, positions are longitude/latitude degrees: lengths
  are great-circle distances times the context's `sphereRadius` (default mean Earth radius,
  `GPU_EARTH_MEAN_RADIUS_METERS`; meters, or kilometers with `units: 'kilometers'`), interior points
  are slerped, and longitudes are unwrapped along each line so it stays continuous across the
  antimeridian. **Breaking:** the `coordinateSystem` and `radius` props of `GPULineSegmentize` and
  `GPULineChunk` (and `radius` of `GPUGreatCircleArcs`) are replaced by `spatialContext`; planar is
  the default. `GPUGreatCircleArcs` defaults to a great-circle longitude/latitude context and
  throws for planar ones; `GPULineSmooth` is planar only and throws for longitude/latitude.
- `status.approximation` (optional, `GPULineSegmentize` and `GPUGreatCircleArcs` only; other contributors that
  write this layout reject a view for it) is set to 1 when an edge or arc needed more than `maximumPieceCount`
  (`maximumSegmentCount`) pieces for the requested maximum length, so some output segments are longer than
  requested; otherwise 0, rewritten on every encoding.
- `GPUGreatCircleArcs` (turf `greatCircle`): one line of `segments + 1` vertices per
  origin/destination pair, `segments = clamp(max(ceil(distance / maximumSegmentLength),
  minimumSegments), 1, maximumSegmentCount)`. Endpoints are exact; longitudes are unwrapped from the
  source (Tokyo to San Francisco ends at 237.6), never split. Draw with `PathLayer`
  `wrapLongitude: true` or in a globe view. The shared interpolator declares an edge unresolved when its float32 central angle
  satisfies `π - angle < 1e-6`. Both interior coordinates are NaN; known endpoints are retained,
  with the target longitude unwrapped from the source. This same rule applies per edge in
  great-circle `GPULineSegmentize`. Supply a caller-selected waypoint to choose the route
  (for example `(0°,0°) → (90°,30°) → (180°,0°)`) and segmentize those two edges. There is no whole-arc rejection: counts and endpoints
  are emitted even when the shared interpolator returns NaN for interior vertices within 1e-6
  radians of an antipode. Exactly antipodal endpoints do not determine a unique great circle.
- `GPULineSmooth` (Chaikin corner cutting, turf `polygonSmooth`): `iterationCount` (compile time,
  1 to 10 passes) each replace edge `(a, b)` with `mix(a, b, r)` and `mix(a, b, 1 - r)` for a per-frame
  ratio `r` (default 0.25). Open lines keep their endpoints (`n` vertices become `n * 2^k`); with
  `closed: true` a repeated closing vertex is ignored and each ring repeats its first vertex.
  Output sizes depend only on the layout, so every level's offsets come from two scans in closed
  form and each level is one kernel; levels ping-pong through two capacity-sized transients.
- `GPULineChunk`: `'chunk'` splits each line into pieces of `chunkLength` (turf `lineChunk`), the
  last piece ending at the line end; consecutive pieces share their boundary point. The piece
  count is data dependent: the line capacity is `output.lineOffsets.length - 1`, and
  `status.count`/`sourceLineIndices` report the pieces; `requiredCount` is the unclamped piece count,
  computed independently of output capacity and saturated at `0xffffffff` when it reaches the uint32
  limit. Through chunk index `2^24`, the piece count is the first index whose rounded float32
  boundary reaches the published line length. Beyond that range, the count uses float32 division
  rounded up.
  Requesting `requiredCount` in chunk mode adds a per-line count scan; layout offsets remain bounded
  even when the full requirement exceeds uint32. `'substring'` extracts `[startMeasure,
  endMeasure]` from every line (turf `lineSliceAlong`, PostGIS `ST_LineSubstring` by distance);
  measures are clamped to the line and a start past the end gives an empty line. A cut at a
  published float32 vertex measure selects the first row with that measure. A cut at the line's
  total measure selects the last row. Other cuts interpolate their position. Each piece retains
  all input rows between its cuts in row order, including coincident vertices and distinct
  vertices whose published measures coincide. Equal substring measures on a positive-length line
  emit the cut position twice. Zero-length lines are copied as one piece. Both modes support a
  great-circle `spatialContext`.
- Everything per frame lives in the parameter buffer, so resolution, ratio, chunk length and
  ranges change with no recompile. All kernels are element-wise or sequential per line:
  deterministic, no float atomics.
- Precision (f32, measured against f64 oracles on Apple Metal): planar densify and chunks within
  2e-5 of the coordinate scale; spherical densify within 2.9 m, arcs up to 16,500 km within 5.7 m.

`getGPULineSegmentizeParameterValues`, `getGPUGreatCircleArcsParameterValues`, and
`getGPULineChunkParameterValues` accept optional `settings.spatialContext`, the context of the
lengths. A lon/lat context warns when a positive finite `maximumSegmentLength` or `chunkLength`
is below ten times the float32 position resolution (about 16.97 meters on the default sphere):
`<ClassName> distance is too short for float32 lon/lat; use createGPULocalFrame`.
Pass the same context used by the contributor. This setting does not change the four-float packed
layout; planar contexts and omitted contexts do not warn.

#### How it computes

{/* lens:GPULineSegmentize */}

::lens{scenario=gpu-spatial-analysis.line-segmentize rung=datum datum=0 caption="Road 0: its vertices before and after 25 m densification"}

::lens{scenario=gpu-spatial-analysis.line-segmentize rung=formula caption="Pieces per segment from its length and the 25 m limit"}

::lens{scenario=gpu-spatial-analysis.line-segmentize rung=kernel caption="The compute shader that writes the densified vertices"}

::lens{scenario=gpu-spatial-analysis.line-segmentize rung=graph caption="Command graph nodes and buffers that produce the densified vertices"}

{/* lens:GPUGreatCircleArcs */}

::lens{scenario=gpu-spatial-analysis.great-circle-arcs rung=datum datum=11 caption="Route 11: ORD to JFK, 33 vertices along the great circle"}

::lens{scenario=gpu-spatial-analysis.great-circle-arcs rung=formula caption="Arc vertices by spherical interpolation at equal angle steps"}

::lens{scenario=gpu-spatial-analysis.great-circle-arcs rung=kernel caption="The compute shaders that count and write the arc vertices"}

::lens{scenario=gpu-spatial-analysis.great-circle-arcs rung=graph caption="Command graph nodes and buffers that produce the arcs"}

{/* lens:GPULineSmooth */}

::lens{scenario=gpu-spatial-analysis.line-smooth rung=datum datum=0 caption="Road 0: corners cut twice at ratio 0.25"}

::lens{scenario=gpu-spatial-analysis.line-smooth rung=formula caption="Each edge is replaced by two points at the cut ratio"}

::lens{scenario=gpu-spatial-analysis.line-smooth rung=kernel caption="The compute shader that writes the smoothed vertices"}

::lens{scenario=gpu-spatial-analysis.line-smooth rung=graph caption="Command graph nodes and buffers that produce the smoothed vertices"}

{/* lens:GPULineChunk */}

::lens{scenario=gpu-spatial-analysis.line-chunk rung=datum datum=0 caption="Piece 0: the first 100 m chunk of road 0"}

::lens{scenario=gpu-spatial-analysis.line-chunk rung=kernel caption="The compute shader that writes the chunk vertices"}

::lens{scenario=gpu-spatial-analysis.line-chunk rung=graph caption="Command graph nodes and buffers that produce the chunk vertices"}

{/* parameters: GPUGreatCircleArcs generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGreatCircleArcs` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUGreatCircleArcs`.

- Packer: `getGPUGreatCircleArcsParameterValues(settings)`, writes `GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUGreatCircleArcsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Context of the lengths; a lon/lat context warns when the length is below float32 resolution. |
| `maximumSegmentLength` | `number` | Largest arc segment length in sphere-radius units (meters by default). `0` or `Infinity` lets `minimumSegments` alone decide the resolution. |
| `minimumSegments` (optional) | `number` | Smallest number of segments per arc. Set `maximumSegmentLength` to `0` and this to `n` for a fixed `n` segments per arc. Default 1. |

{/* /parameters */}

{/* parameters: GPULineChunk generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULineChunk` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULineChunk`.

- Packer: `getGPULineChunkParameterValues(settings?)`, writes `GPU_LINE_CHUNK_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPULineChunkParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Context of the lengths; a lon/lat context warns when the length is below float32 resolution. |
| `chunkLength` (optional) | `number` | `'chunk'` mode: piece length in position (or sphere-radius) units. |
| `startMeasure` (optional) | `number` | `'substring'` mode: start distance from each line's first vertex, clamped to the line. |
| `endMeasure` (optional) | `number` | `'substring'` mode: end distance, clamped to the line. A start past the end gives an empty line. |

{/* /parameters */}

{/* parameters: GPULineSegmentize generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULineSegmentize` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULineSegmentize`.

- Packer: `getGPULineSegmentizeParameterValues(settings)`, writes `GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPULineSegmentizeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Context of the lengths; a lon/lat context warns when the length is below float32 resolution. |
| `maximumSegmentLength` | `number` | Largest output segment length, in position units (`'planar'`) or sphere-radius units (`'spherical'`, meters by default). `0` or `Infinity` disables densification. |

{/* /parameters */}

{/* parameters: GPULineSmooth generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULineSmooth` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULineSmooth`.

- Packer: `getGPULineSmoothParameterValues(settings?)`, writes `GPU_LINE_SMOOTH_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPULineSmoothParameterValues`:

| Field | Type | Description |
|---|---|---|
| `ratio` (optional) | `number` | Chaikin cut ratio in `(0, 0.5]`. Each edge `(a, b)` is replaced by `mix(a, b, ratio)` and `mix(a, b, 1 - ratio)`. Default 0.25 (classic Chaikin). |

{/* /parameters */}

### `GPUGeometryMeasures`, `GPUGeodesicPairs`, and `GPUGeodesicDestination`

When requesting per-feature or per-group lengths or areas, GPUGeometryMeasures rejects known
angular planar frames even with metric none and omitted result units. Project into a planar
linear frame first. Coordinate-only columns with metric none remain usable; linear frames
retain their declared units, and unlabelled XY cannot establish ground units.

The [geodesic precision guide](geodesic-precision.md) defines per-row status, conditioning
exclusions, measured error envelopes and differences from the float64 CPU operation twins.

**`GPUGeometryMeasures`** — Measures length, area, centroid, bounding box and vertex count of line and polygon features, planar or on the ellipsoid.

::lens{scenario=gpu-spatial-analysis.geometry-measures rung=result caption="Streets coloured by WGS84 geodesic length in metres, beside the 223 Wicker Park streets"}

**`GPUGeodesicPairs`** — Computes distance, initial and final bearing and midpoint for each origin and target pair of longitude/latitude points.

::lens{scenario=gpu-spatial-analysis.geodesic-pairs rung=result caption="Grey: 40 ORD destinations · blue: route midpoints; distances are WGS84 Vincenty"}

**`GPUGeodesicDestination`** — Projects each origin along a bearing and distance, writing the destination point and final bearing.

::lens{scenario=gpu-spatial-analysis.geodesic-destination rung=result caption="Grey: ORD · blue: destinations at 8 bearings and 500, 1,000 and 2,000 km (WGS84)"}

Measurement columns for line and polygon features and for point pairs, in planar coordinates or
on the Earth (turf `area`, `length`, `centroid`, `bbox`, `distance`, `bearing`, `destination`,
`midpoint`; PostGIS `ST_Area`, `ST_Length`, `ST_Perimeter`, `ST_Centroid`, `ST_Envelope`,
`ST_NPoints`, `ST_Distance(geography)`, `ST_Azimuth`, `ST_Project`).

`GPUGeometryMeasures.spatialContext` is optional (native planar by default). `GPUGeodesicPairs` and
`GPUGeodesicDestination` require a longitude/latitude context and declare `measure: 'distance'`;
geometry measures declare no measure because they output both lengths and areas. All three use
`resolveGPUSpatialContext` and its shared validation errors.

```ts
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUGeometryMeasures, GPUGeodesicPairs, GPUGeodesicDestination} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample1(graph: GPUCommandGraph, inputs: {
  positions: NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['positions']>;
  featureOffsets: NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['featureOffsets']>;
  polygonOffsets: NonNullable<Extract<ConstructorParameters<typeof GPUGeometryMeasures>[0], {geometryType: 'polygons'}>['polygonOffsets']>;
  ringOffsets: NonNullable<Extract<ConstructorParameters<typeof GPUGeometryMeasures>[0], {geometryType: 'polygons'}>['ringOffsets']>;
  areas: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['output']>['areas']>;
  lengths: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['output']>['lengths']>;
  centroids: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['output']>['centroids']>;
  bounds: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['output']>['bounds']>;
  vertexCounts: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['output']>['vertexCounts']>;
  invalidCount: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['output']>['status']>['invalidCount']>;
  groupIds: NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['groupIds']>;
  groupAreas: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['groupOutput']>['areas']>;
  groupCentroids: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['groupOutput']>['centroids']>;
  featureCounts: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeometryMeasures>[0]['groupOutput']>['featureCounts']>;
  origins: NonNullable<ConstructorParameters<typeof GPUGeodesicPairs>[0]['origins']>;
  targets: NonNullable<ConstructorParameters<typeof GPUGeodesicPairs>[0]['targets']>;
  distances: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeodesicPairs>[0]['output']>['distances']>;
  initialBearings: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeodesicPairs>[0]['output']>['initialBearings']>;
  finalBearings: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeodesicPairs>[0]['output']>['finalBearings']>;
  midpoints: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeodesicPairs>[0]['output']>['midpoints']>;
  converged: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUGeodesicPairs>[0]['output']>['status']>['converged']>;
  bearings: NonNullable<ConstructorParameters<typeof GPUGeodesicDestination>[0]['bearings']>;
  destinations: NonNullable<NonNullable<ConstructorParameters<typeof GPUGeodesicDestination>[0]['output']>['destinations']>;
}) {
  const {positions, featureOffsets, polygonOffsets, ringOffsets, areas, lengths, centroids, bounds, vertexCounts, invalidCount, groupIds, groupAreas, groupCentroids, featureCounts, origins, targets,  distances, initialBearings, finalBearings, midpoints, converged, bearings, destinations} = inputs;
  graph.add(new GPUGeometryMeasures({
    geometryType: 'polygons',
    // GeoArrow four-level layout: feature -> polygons -> rings -> vertices
    positions, featureOffsets, polygonOffsets, ringOffsets,
    // planar (native units), great-circle (spherical), or ellipsoidal (geodesic edges by default)
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal', units: 'meters'},
    output: {
      areas, lengths, centroids, bounds, vertexCounts,
      status: {invalidCount} // geographic polygons with a pole or half-world edge, see below
    },
    groupIds, groupCount: 12,
    groupOutput: {areas: groupAreas, centroids: groupCentroids, featureCounts}
  }));

  const wgs84 = {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal', units: 'meters'} as const;
  graph.add(new GPUGeodesicPairs({origins, targets, spatialContext: wgs84,
    output: {distances, initialBearings, finalBearings, midpoints, status: {converged}}}));
  graph.add(new GPUGeodesicDestination({origins, bearings, distances, spatialContext: wgs84,
    output: {destinations, finalBearings}}));
}
```

- Features: one invocation per feature walks its rings in row order with Neumaier-compensated f32
  sums, relative to the feature's first vertex, so projected coordinates far from the origin keep
  their precision (a 1 m square at x = 1e6 measures exactly 1). Rings close implicitly; a repeated
  closing vertex adds a zero edge. Outputs: length (lines) or perimeter (polygons), signed and
  absolute area, area-weighted (polygons) or length-weighted (lines) centroid falling back to the
  vertex mean, bounds, vertex count. Empty features give zeros and NaN centroid and bounds.
- Input layout: `geometryType` selects the layout. `'polygons'` takes the four-level layout of
  `GPUPolygonGeometryFields` (`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`; rows are
  features). `'lines'` takes `positions` and `lineOffsets`, plus optional `featureOffsets`
  (feature -> lines) for multi-lines. `'points'` takes `positions`, plus optional `featureOffsets`
  (feature -> points) for multi-points; without it every point is a feature.
- Ring roles are positional: the first ring of every polygon is a shell and later rings are holes
  whatever the winding, so a multi-polygon with mixed winding and a hole wound like its shell measure
  correctly, and `areas` is non-negative. `GPUShapeDescriptors`, `GPUPolygonTriangulation` and
  `GPUAffineTransform` (centroid origins) use the same layout and rule; `GPULabelPoint` uses even-odd,
  which agrees with it for valid (non-overlapping) polygons.
- Point and line `areas` are exact zero, including group totals. Planar bounds and extreme rows
  cover every finite float32 coordinate, including both signs of `FLT_MAX`; group reductions
  use the same full range. Bounds use finite sentinels; extreme-coordinate ties select the
  lowest original row, including coordinates equal to those sentinels. Unsupported requested measure columns (for example `signedAreas`)
  are rejected before descriptor snapshotting can discard them. Planar bounds and extreme rows
  compare original float32 coordinates, preserving small distinctions far from the moment origin.
  Local coordinates are still used for moments. When a multiline's total length is zero, its
  centroid averages one point per nonempty collapsed part; repeated vertices add no extra weight.
- With `metric: 'great-circle'` and linear edges (the default), area uses Chamberlain-Duquette
  (turf parity): an approximation of degree-straight edges with second-order error in latitude
  span, exact for longitude/latitude boxes. With `edgeModel: 'great-circle'`, area uses signed
  spherical excess on the context sphere (`areaModel: 'sphere-polygon'`), summed per edge as the
  Lambert chord term from the feature origin less the bulge of the great-circle arc over that
  chord, so a ring a hundred meters across keeps the same f32 relative precision (about `5e-7`)
  as a continental one. Perimeter uses
  great-circle chords for both edge models; centroids retain the straight Lambert equal-area
  construction. For linear-edge area, pass
  `sphereRadius: GPU_WGS84_SEMI_MAJOR_AXIS_METERS` for turf `area` parity. Ellipsoidal with
  `ellipsoidalEdgeModel: 'coordinate-linear'` maps latitude to authalic latitude (exact equal-area
  ellipsoid mapping) and measures edges with an f32 Vincenty inverse.
- **Measure model.** The instance's `measures` field (`GPUMeasureDescription`) says what the length
  and area outputs mean: `lengthModel` (`'planar'`, `'great-circle'`, `'ellipsoidal-geodesic'`),
  `areaModel` (`'planar'`, `'sphere-equal-area'`, `'sphere-polygon'`, `'wgs84-authalic'`, `'geodesic'`) and the length
  and area units. Units follow `getGPULengthUnits`: native planar uses position units; a meters local
  frame reports meters and square meters. Geographic lengths default to meters and areas to square
  meters; `units: 'kilometers'` gives kilometers and square kilometers, including ellipsoidal kernels
  (previously rejected). Feature and group areas use the context's length unit squared.
- Linear-edge spherical areas use `getGPUAuthalicSphereRadius` (`areaRadius`); great-circle-edge
  areas use `getGPUSphereRadius`. Lengths keep the
  mean radius (`radius`). With the default sphere, area results change by about `5e-7` relative.
  Ellipsoidal kernels honor `spatialContext.ellipsoid` (default WGS84); the geodesic classes expose the
  resolved `ellipsoid` in the context's length unit.
- **Antimeridian and poles (geographic polygons).** Non-pole longitude steps are wrapped, so an
  edge with a longitude difference above 180 degrees (Fiji's 179 to -179) takes the short way and
  the ring is measured correctly. Same-pole edges retain their raw step, so explicitly capped
  Antarctica rings have zero winding and finite area. Unprepared rings with winding +-360,
  rings wider than 360 degrees, and ambiguous edges report NaN `areas` and
  `centroids` (lengths, bounds and vertex counts stay valid) and are counted in
  `output.status.invalidCount`, a one-row `uint32` view cleared every encoding. Linear edges with
  a wrapped step of 180 degrees are ambiguous; great-circle edges are ambiguous when antipodal
  or passing through a pole. Prepare global rings with `GPULonLatPolygonPreparation`.
  Ellipsoidal contexts take `ellipsoidalEdgeModel`; declaring `edgeModel` throws.
  The check covers the serial and cooperative
  kernels. Line features are unaffected.
- Groups: features are stably sorted by group ID and one workgroup per group reduces them with a
  fixed tree, so group sums are bitwise reproducible. Group centroids are area- (or length-)
  weighted means of feature centroids. **Behavior change:** in geographic contexts the mean is
  taken on 3D unit vectors (a spherical centroid), not on longitude and latitude numbers, so a
  group straddling the antimeridian lands on the antimeridian (for example 179 and -179 give 180)
  instead of at longitude 0. An antipodal group has no mean direction and reports NaN; a group
  containing an invalid feature reports NaN. Planar groups are unchanged.
  Feature centroid longitude stays on that feature's unwrapped branch; geographic group centroid
  longitude uses the principal `[-180,180]` branch. Geographic group bounds translate whole feature
  intervals by whole turns to align their centers with the first nonempty interval's center,
  in stable feature-row order, before taking the union. Already-contained duplicate endpoints
  remain contained. Compare bounds and a viewport on a common longitude branch: translate the
  whole viewport or interval by multiples of 360°, or union the masks for each visible turn.
  Do not independently wrap the interval endpoints or the feature centroid.
- `'geodesic'` (`pyproj.Geod.geometry_area_perimeter`, geo `GeodesicArea`): lengths and perimeters are
  Vincenty geodesic lengths, as `'wgs84'`, and polygon areas follow geodesic edges on the context's
  ellipsoid (WGS84 by default). Edges of 20 km or more are sampled along the geodesic (spans of at most 250 km, at most 16)
  and summed as a chord plus the exact parabolic-segment bulge in local equal-area coordinates, so f32
  never subtracts two huge areas. Within 1e-6 relative of pyproj on regional and continental polygons
  (perimeter 8e-6), where `'wgs84'` straight edges are off by up to 30% on very long edges. Centroids
  still use straight equal-area edges, edges where Vincenty does not converge stay straight, and edges
  under 20 km ignore the bulge.
- `output.extremeVertices` (`uint32x4`): per feature `[argMinX, argMinY, argMaxX, argMaxY]` as global
  `positions` rows, lowest row on ties, `0xffffffff` for empty features. Geographic x is the unwrapped
  longitude of `bounds`. Per feature only (rejected in `groupOutput`).
- `geometryType: 'points'`: every feature is one point or a multi-point. Outputs are lengths (0), centroids
  (vertex mean, as Shapely `MultiPoint.centroid`), bounds, vertex counts and `extremeVertices`;
  `areas` publishes zero per feature and group. Group centroids weight by vertex count.
- `GPUGeodesicPairs`: distance, initial and final bearing (degrees clockwise from north in
  `(-180, 180]`), and midpoint per pair. `'sphere'` uses a Vincenty-form central angle that is well
  conditioned from millimeters to antipodal; `'wgs84'` runs bounded Vincenty inverse iteration.
  See the [precision guide](geodesic-precision.md#gpugeodesicpairs) for the iteration limits,
  conditioning exclusions, fallback status and CPU twin. luma.gl's
  `GPUHaversineDistance` computes the same sphere distance with a
  default radius of 6371 (kilometers); this class takes radius and unit from the spatial context
  (6371008.8 m by default) and takes longitude differences on the seam side, so short pairs across
  the antimeridian keep their precision in every model.
  Pair outputs, including status views, must not overlap one another or either input view;
  disjoint slices of one buffer are allowed and checked at construction.
- `GPUGeodesicDestination`: destination and final bearing from origin, bearing and distance
  columns, on the sphere or by Vincenty's direct solution.
  No warning is possible because the distance is a GPU view; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).
- `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb', units: 'meters'}`
  on both classes (turf `rhumbDistance`, `rhumbBearing`, `rhumbDestination`; geo `Rhumb`):
  a line of constant bearing on the context sphere (default radius 6371008.8 m).
  Coordinates (`origins`, `targets`, `midpoints`, `destinations`) are longitude/latitude degrees;
  `bearings`, `initialBearings` and `finalBearings` are degrees clockwise from north.
  Distance inputs and outputs are meters in this context. Pairs give `distances`,
  `initialBearings` equal to `finalBearings` (constant, in `(-180, 180]`) and the rhumb
  `midpoints` (the midpoint by distance: the mean latitude, and the longitude of the loxodrome at
  that latitude, which is linear in the Mercator latitude and differs from the mean longitude
  unless the line runs east-west). Coincident pairs have zero distance and zero initial and final
  bearings. The destination longitude is continuous with the origin, and `finalBearings` flips to the
  opposite north-south sense when a move overshoots a pole and is reflected, as turf does (the reflected
  case is approximate). Longitude differences take the short way across the antimeridian, latitudes clamp
  at 89.9999 degrees, equal-latitude edges use the east-west limit `cos(lat)`, and the Mercator-stretched
  latitude difference is an `atanh` of a cancellation-free ratio. Sphere only (turf and geo are spherical
  too), `status.converged` is ellipsoidal-only, and the metric switch is compile-time.
  For `(0°,60°) → (90°,60°)`, the closed-form rhumb distance is `R·π/4 = 5003778.610508981 m`,
  versus great-circle `R·acos(3/4) = 4604546.2528806515 m` with `R=6371008.8 m`.
  A rhumb destination from `(0°,60°)` at bearing 90° and distance `5003778.610508981 m`
  returns `(90°,60°)` (subject to f32 rounding). For the ellipsoidal route, use
  `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal', units: 'meters'}`
  as in the example above. Distance error against an f64
  closed form was 0.74 m (6e-8 relative) over 12,108 km and bearings were within 1e-4 degrees. Rhumb length
  in `GPUGeometryMeasures` and rhumb densify in `GPULineSegmentize` are not provided.
- Geometry-measure precision (f32 vs f64 oracles, Apple Metal): WGS84 and spherical box areas
  within 3e-7 relative from 100 m blocks to 2,000 km boxes. Geodesic-pair and destination bounds
  and the distinction between nominal arithmetic measurements and GPU certification are in the
  [precision guide](geodesic-precision.md#measurements-and-acceptance-bounds).
  See [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions) for input quantization.

#### How it computes

{/* lens:GPUGeometryMeasures */}

::lens{scenario=gpu-spatial-analysis.geometry-measures rung=datum datum=0 caption="Road 0: WGS84 geodesic length in metres"}

::lens{scenario=gpu-spatial-analysis.geometry-measures rung=formula caption="Road length as the sum of geodesic edge distances"}

::lens{scenario=gpu-spatial-analysis.geometry-measures rung=kernel caption="The compute shader that writes the road lengths"}

::lens{scenario=gpu-spatial-analysis.geometry-measures rung=graph caption="Command graph nodes and buffers that produce the road lengths"}

{/* lens:GPUGeodesicPairs */}

::lens{scenario=gpu-spatial-analysis.geodesic-pairs rung=datum datum=15 caption="Route 15: ORD to LHR distance and initial bearing"}

::lens{scenario=gpu-spatial-analysis.geodesic-pairs rung=kernel caption="The compute shader that runs the capped Vincenty inverse per pair"}

::lens{scenario=gpu-spatial-analysis.geodesic-pairs rung=graph caption="Command graph nodes and buffers that produce the pair columns"}

{/* lens:GPUGeodesicDestination */}

::lens{scenario=gpu-spatial-analysis.geodesic-destination rung=datum datum=0 caption="Row 0: the destination 500 km due north of ORD"}

::lens{scenario=gpu-spatial-analysis.geodesic-destination rung=kernel caption="The compute shader that runs the capped Vincenty direct per row"}

::lens{scenario=gpu-spatial-analysis.geodesic-destination rung=graph caption="Command graph nodes and buffers that produce the destinations"}

### `GEODESIC_WGSL`, `createGPULocalFrame`, and `GPULocalFrameProjection`

**`GPULocalFrameProjection`** — Projects longitude/latitude positions on the GPU into a local azimuthal-equidistant frame in meters.

::lens{scenario=gpu-spatial-analysis.local-frame-projection rung=result caption="Blue: 223 Wicker Park streets projected from longitude/latitude to local metres"}

Two ways to work on longitude/latitude data, used consistently across the package:

- **Measure on the sphere.** `GEODESIC_WGSL` is the public WGSL source of the spherical functions
  every great-circle path in this package uses; include it once in your own compute shader.
  Positions are `vec2<f32>(longitude, latitude)` degrees and distances are in the units of the
  `radius` argument (`GPU_EARTH_MEAN_RADIUS_METERS` for meters, the package default).
  `geodesicHaversineDistance(a, b, radius)` is the great-circle distance;
  `geodesicSegmentFraction(point, start, end)` and
  `geodesicSegmentDistance(point, start, end, fraction, radius)` give the point-to-segment distance
  for segments that are straight in longitude/latitude (the edge model of the containment tests and
  network edges; exact to first order for kilometer-length segments, densify longer ones). Projection
  and interpolation follow the same wrapped start-to-end longitude arc. The fraction helper compares
  its projected candidate with both endpoints, so its distance cannot exceed either endpoint distance.
  An interior projection survives a rounded distance tie; a tie between endpoint candidates selects
  the start. CPU line simplification uses the same wrapped arc and endpoint comparison as the GPU.
  Local GPU candidates are ordered by their common haversine terms before inverse trigonometry,
  so different `asin`/`atan2` rounding cannot displace a local interior projection. Distant candidates
  keep central-angle comparisons to preserve resolution near antipodes.
  The distance helper evaluates the supplied fraction; it does not select a new fraction. Also
  available: `geodesicHaversineTerm`, `geodesicCentralAngle`, `geodesicInitialBearingDegrees`,
  `geodesicDestination`, `geodesicInterpolate` and `geodesicWrapLongitudeDelta`.
- **Project first.** Planar operators throw on longitude/latitude contexts. `createGPULocalFrame({origin,
  kind})` returns a binary64 CPU frame in meters: `'azimuthal-equidistant'` (spherical, mean radius;
  distances and bearings from the origin exact) or `'lambert-equal-area'` (WGS84 ellipsoidal, areas
  exact), matching PROJ `aeqd +R=6371008.8` and `laea +ellps=WGS84` to a millimeter. It offers
  `project(lng, lat)`, `unproject(x, y)` and `projectArray(lngLats, stride)` for uploads, and a
  `spatialContext` that carries the frame identity: contexts of frames with different origins are
  incompatible, so data projected around different origins is not mixed. Float32 frame coordinates
  keep about 1e-7 of the frame extent (millimeters at 10 km).
  See [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions) for float32 degrees.
  `GPULocalFrameProjection` projects GPU-resident `float32x2` or binary64
  (`uint32x4`) longitude/latitude rows into the frame without a readback (compiled adaptive patches
  from `gpu-project`, default tolerance 1 cm, output in absolute frame meters). Retain the
  contributor to release its projection plan, including when graph declaration or compilation fails:

```ts
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPULocalFrameProjection, type GPULocalFrameProjectionProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

export async function withLocalFrameProjection(
  graph: GPUCommandGraph,
  props: GPULocalFrameProjectionProps,
  useCompiled: (compiled: ReturnType<typeof graph.compile>) => void | Promise<void>
) {
  const projection = new GPULocalFrameProjection(props);
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    graph.add(projection);
    compiled = graph.compile();
    await useCompiled(compiled); // Submit and consume the graph here.
  } finally {
    try {
      compiled?.destroy();
    } finally {
      projection.destroy();
    }
  }
}
```

  The awaited callback keeps both owners alive until graph use finishes.
  The frame's `spatialContext` is `{coordinateSpace: 'planar', metric: 'native', units: 'meters', frame}`.
  Choose `'azimuthal-equidistant'` for distance operators and `'lambert-equal-area'` for area/density
  operators; the wrong frame kind throws at operator construction. `GPULocalFrameProjection.spatialContext`
  is the frame's context.

#### How it computes

{/* lens:GPULocalFrameProjection */}

::lens{scenario=gpu-spatial-analysis.local-frame-projection rung=datum datum=0 caption="Vertex 0: its longitude/latitude and projected frame metres"}

::lens{scenario=gpu-spatial-analysis.local-frame-projection rung=formula caption="Local polynomial patch maps normalised lon/lat to frame metres"}

::lens{scenario=gpu-spatial-analysis.local-frame-projection rung=kernel caption="The compute shader that writes the projected positions"}

::lens{scenario=gpu-spatial-analysis.local-frame-projection rung=graph caption="Command graph nodes and buffers that produce the projected positions"}

### `GPULinearReferencing` and `GPULineLocate`

**`GPULinearReferencing`** — Snaps points to their nearest polyline within a radius, reporting the road, distance along it and foot point.

::lens{scenario=gpu-spatial-analysis.linear-referencing rung=result caption="Grey: 223 Wicker Park streets · blue: foot points of 40 Overture places within 500 m"}

**`GPULineLocate`** — Places events on polylines by measure, writing the located point and a status code per event. A measure outside the line is clamped to the nearest end (status `clamped`); a NaN measure, an empty line or a line index out of range is `invalid` with NaN outputs.

::lens{scenario=gpu-spatial-analysis.line-locate rung=result caption="Grey: 223 Wicker Park streets · blue: 40 Overture places located by their measure"}

For `GPULinearReferencing.radius`, no warning is possible because the distance is a GPU view; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

Maxima (functions of the device, default-device values for 128 MiB bindings in parentheses): `GPULinearReferencing` takes at most `getGPULinearReferencingMaximumQueryCount(device)` (3,728,270) `queryPositions` and `positions` of at most `getGPULinearReferencingMaximumPlanarPositionCount(device)` (2^23, planar BVH) or `getGPULinearReferencingMaximumSphericalPositionCount(device)` (2^21, arc table). `GPULineLocate` takes at most `getGPULineLocateMaximumEventCount(device)` (5,592,405) events.

Linear referencing on many polylines (`positions` sorted by path, `lineOffsets`): snap points to
the nearest path and report where they are along it, or place events on paths by measure.

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPULinearReferencing, GPULineLocate, getGPULineLocateParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample2(graph: GPUCommandGraph, inputs: {
  queryPositions: NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['queryPositions']>;
  positions: NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['positions']>;
  lineOffsets: NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['lineOffsets']>;
  radiusParameters: GPUParameterBuffer<'float32'>;
  lineIndices: NonNullable<NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['output']>['lineIndices']>;
  segmentIndices: NonNullable<NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['output']>['segmentIndices']>;
  fractions: NonNullable<NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['output']>['fractions']>;
  footPoints: NonNullable<NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['output']>['footPoints']>;
  distances: NonNullable<NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['output']>['distances']>;
  measures: NonNullable<NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['output']>['measures']>;
  sides: NonNullable<NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['output']>['sides']>;
  signedOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['output']>['signedOffsets']>;
  candidateOverflow: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['output']>['status']>['candidateOverflow']>;
  requiredCandidateCount: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULinearReferencing>[0]['output']>['status']>['requiredCandidateCount']>;
  eventLineIndices: NonNullable<ConstructorParameters<typeof GPULineLocate>[0]['eventLineIndices']>;
  eventMeasures: NonNullable<ConstructorParameters<typeof GPULineLocate>[0]['eventMeasures']>;
  eventOffsets: NonNullable<ConstructorParameters<typeof GPULineLocate>[0]['eventOffsets']>;
  locateParameters: GPUParameterBuffer<'float32'>;
  eventPositions: NonNullable<NonNullable<ConstructorParameters<typeof GPULineLocate>[0]['output']>['positions']>;
  angles: NonNullable<NonNullable<ConstructorParameters<typeof GPULineLocate>[0]['output']>['angles']>;
  statuses: NonNullable<NonNullable<ConstructorParameters<typeof GPULineLocate>[0]['output']>['statuses']>;
  elapsedMeters: number;
}) {
  const {queryPositions, positions, lineOffsets, radiusParameters, lineIndices, segmentIndices, fractions, footPoints, distances, measures, sides, signedOffsets, candidateOverflow, requiredCandidateCount, eventLineIndices, eventMeasures, eventOffsets, locateParameters, eventPositions, angles, statuses, elapsedMeters} = inputs;
  graph.add(new GPULinearReferencing({
    queryPositions, positions, lineOffsets,
    radius: radiusParameters.importToGraph(graph), // one float32 row, per frame
    candidateCapacity: 1 << 20, spatialSort: true,
    output: {lineIndices, segmentIndices, fractions, footPoints, distances, measures, sides, signedOffsets,
             status: {candidateOverflow, requiredCandidateCount}}
  }));

  graph.add(new GPULineLocate({
    positions, lineOffsets, eventLineIndices, eventMeasures, eventOffsets,
    measureMode: 'distance', // or 'fraction'
    parameters: locateParameters.importToGraph(graph),
    output: {positions: eventPositions, angles, statuses}
  }));
  locateParameters.write(getGPULineLocateParameterValues({measureOffset: elapsedMeters})); // animate
}
```

- `GPULinearReferencing` (turf `nearestPointOnLine`, PostGIS `ST_ClosestPoint` and
  `ST_LineLocatePoint`): every vertex row becomes a segment to the next row of its line and
  `GPUNearestFeatureJoin` finds each point's nearest segment within the radius through its BVH.
  Ties go to the smallest segment row (earliest along the line, lowest line). Outputs per point:
  line, segment within the line, fraction along the segment, foot point, distance, measure (distance
  along the line to the foot point), side (`1` left of the line direction, `-1` right, `0` on it)
  and `side * distance`. Points with nothing in range get `0xffffffff` indices, distance `-1`,
  side `0` and NaN elsewhere. Both capacities (BVH leaves and, planar, the bounding-box candidates)
  are intermediate: the required `output.status.candidateOverflow` is 1 when either was too small,
  so some points may have missed their nearest line, and the optional
  `output.status.requiredCandidateCount` is the unclamped candidate count (planar; 0 for a
  great-circle context), so `candidateCapacity` of at least that value removes the candidate cause.
  There is no `count` and no `overflow`: every row is one query point. **Breaking (2026-11-10):**
  `status.overflow` (the same flag) and `status.candidateCount` are removed, and the unclamped
  candidate count moved from `status.requiredCount` to `status.requiredCandidateCount`. `queryPositions` are the points to snap; `positions` and
  `lineOffsets` are the lines.
- `GPULineLocate` (turf `along`, PostGIS `ST_LineInterpolatePoint` and `ST_LocateAlong`): each
  event's measure (distance or fraction, optionally `measure * scale + offset` from the parameter
  buffer) is clamped to the line (`statuses` 1) and located by binary search over cumulative vertex
  measures, then moved by an optional lateral offset along the left normal. Outputs: position,
  segment, unit tangent, angle (degrees counter-clockwise from +x, the deck.gl `getAngle`
  convention) and status (`2` for empty or out-of-range lines, NaN position).
- Cumulative measures are a per-line Neumaier-compensated f32 prefix (optionally output as
  `output.vertexMeasures`). Planar by default: project geographic data first, or use `'spherical'` below.
  In the cooperative route, each published row combines the incoming carry and that row's tile
  prefix with compensation, then combines the resulting sum and correction for the output.
  Compensation applies at publication, including the last moving endpoint, rather than only at
  tile boundaries. For 64 steps of length 16 followed by 128 steps of length `2^-20`, that endpoint
  measure is exactly `1024.0001220703125`. `GPULineChunk` therefore needs two chunks at chunk
    length 1024. A published measure can plateau between distinct moving vertices: rows 191 and
  192 both round to that total, but their positions differ. Chunk and substring cuts resolve
  interior ties at the first published row attaining the cut, consistently for adjoining pieces.
  Start/end measures preserve known input endpoints; a zero-width terminal substring emits
  that endpoint twice. Projection at segment fraction one reads the published next-row measure;
  normalized measure one and location at the total retain the terminal endpoint.
  `GPULinearReferencing`'s `output.signedOffsets` are physical signed cross-track distances in the context's
  length unit; they are distinct from `lineOffsets`, the integer row boundaries of the line layout.
- `output.normalizedMeasures` (both coordinate systems) is `measures / lineLength`, 0 for zero-length
  lines and NaN when unmatched (Shapely `line_locate_point(normalized=True)`).
  `GPULineLocate({measureMode: 'fraction'})` is `line_interpolate_point(normalized=True)`.
- `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` on both classes
  (planar is the default): positions are longitude and latitude in degrees and segments are
  great-circle arcs. `distances`, `measures`, `signedOffsets` and the search `radius` use the context's
  length unit (meters by default, kilometers with `units: 'kilometers'`) on its `sphereRadius`
  (default mean Earth radius, 6371008.8 meters; `radius` already names the search radius).
  Both declare `measure: 'distance'`: a Lambert equal-area local frame throws; use an azimuthal-equidistant frame.
  **Breaking:** the `coordinateSystem` and `sphereRadius` props are removed in favor of
  `spatialContext`; other contexts throw.
  The foot point is the closest point on the arc (geo `HaversineClosestPoint`), with a longitude that is
  continuous with the segment start, so it can exceed 180 across the antimeridian. `distances` is the
  cross-track distance and `measures` the along-track measure. `GPULineLocate` slerps along the arc
  (`Geod.fwd`, `npts`): `tangents` are `(east, north)` compass components, offsets move along the left
  compass bearing in the context's length unit and `angles` are counter-clockwise from east. The nearest-arc search walks
  a `GPUBVH` over conservative longitude/latitude boxes of the arcs (the great circle's latitude peak
  included; arcs across the antimeridian or near a pole span every longitude), Hilbert sorted from 256
  positions (`spatialSort`), branch-and-bound with a great-circle box lower bound and exact arc
  distances at the leaves, ties to the lowest segment row. It needs no `candidateCapacity`;
  `output.status.overflow` is 1 only when `leafCapacity` is below the position count. Arcs over about
  179 degrees are unsupported. Against `pyproj.Geod(a=6371008.8,
  f=0)` the worst error was 1.5 m for distance, 0.8 m for measure and 1.7 m for the foot point.

#### How it computes

{/* lens:GPULinearReferencing */}

::lens{scenario=gpu-spatial-analysis.linear-referencing rung=datum datum=0 caption="Place 0: nearest road, distance along it and distance to it"}

::lens{scenario=gpu-spatial-analysis.linear-referencing rung=formula caption="Measure along the road: prefix length plus fraction of the nearest segment"}

::lens{scenario=gpu-spatial-analysis.linear-referencing rung=kernel caption="The compute shader that writes the foot points and measures"}

::lens{scenario=gpu-spatial-analysis.linear-referencing rung=graph caption="Command graph nodes and buffers that produce the foot points and measures"}

{/* lens:GPULineLocate */}

::lens{scenario=gpu-spatial-analysis.line-locate rung=datum datum=0 caption="Place 0: nearest road, measure along it and status"}

::lens{scenario=gpu-spatial-analysis.line-locate rung=formula caption="Event measure becomes a segment fraction, then a point on that segment"}

::lens{scenario=gpu-spatial-analysis.line-locate rung=kernel caption="The compute shader that writes the located points"}

::lens{scenario=gpu-spatial-analysis.line-locate rung=graph caption="Command graph nodes and buffers that produce the located points"}

{/* parameters: GPULineLocate generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULineLocate` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULineLocate`.

- Packer: `getGPULineLocateParameterValues(settings?)`, writes `GPU_LINE_LOCATE_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPULineLocateParameterValues`:

| Field | Type | Description |
|---|---|---|
| `measureScale` (optional) | `number` | Multiplies every event measure before locating. Default 1. |
| `measureOffset` (optional) | `number` | Added to every scaled event measure, for example elapsed distance in an animation. Default 0. |

{/* /parameters */}

### `GPULineSplit` and `GPULineMerge`

**`GPULineSplit`** — Splits polylines at their mutual intersections, writing the pieces with their source-line indices.

Optional `lines.featureOffsets` groups line parts into logical features, including empty features.
Each part splits independently at all intersections; no connector is inserted between parts.
Optional `output.pieces.geometry.featureOffsets` publishes feature-to-piece offsets with the
input feature count plus one rows. Truncation leaves empty tail features visible. Optional
`pieces.sourceIds` publishes the logical feature's caller ID, or feature row when no IDs are
supplied; optional `pieces.sourceFeatureIds` publishes the same identity. Use
`pieces.sourcePartRows` for original input line-part rows and `pieces.sourceFeatureRows` for
logical feature rows. Unused provenance rows are `0xffffffff`. Piece/vertex counts and
capacities continue to count parts and stored vertices.
The terminal feature offset is the active piece count. Remaining allocated line offsets repeat
the terminal vertex offset; CPU track and loop-removal consumers validate and accept that padding.

`GPULineSplit` and `GPULineClipByPolygon` reject `pieces.geometry.sourceIds`: caller identities
belong in `pieces.sourceIds` (or optional `pieces.sourceFeatureIds`). Their piece status supports `count`, `overflow`,
`candidateOverflow`, `requiredCount`, `vertexCount` and `requiredVertexCount` only.
The inherited `complete`, `unresolvedCount`, `uncertainCount`, `invalidCount`,
iteration/convergence/approximation fields, `ringCount` and `requiredRingCount` are rejected because
these contributors do not publish those
piece diagnostics. Use top-level `output.status.uncertainCount` for uncertifiable predicates and
`pieces.status.overflow` for incomplete output.

::lens{scenario=gpu-spatial-analysis.line-split rung=result caption="Grey: 223 Wicker Park streets · blue: pieces split at street intersections"}

**`GPULineMerge`** — Merges polylines that meet end to end at degree-two nodes into longer chains.

::lens{scenario=gpu-spatial-analysis.line-merge rung=result caption="Grey: 223 Wicker Park streets · blue: chains merged through degree-two endpoints"}

Maximum (`GPULineSplit`): `lines.positions` has at most `getGPUSegmentIntersectionMaximumVertexCount(device)` (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows, and the line count plus four times `intersectionCapacity` is at most `getGPULineSplitMaximumPieceBound(device)` (4194304 on a default device).

`GPULineSplit` is planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first. `GPULineMerge` also accepts a longitude-latitude `spatialContext`: lines join only where end points have identical coordinates (bit patterns), nothing is measured, and longitude 180 and -180 are different coordinates.

`GPULineSplit` splits linestrings at every crossing, touch, shared vertex and overlap end (turf `lineSplit`
against the union of the lines; verified against Shapely `unary_union`). It reuses `GPUSegmentIntersection`
in self mode, groups the split events by segment with a stable `GPUSort`, removes duplicates and writes
GeoArrow-layout pieces. Props: `lines` (`{kind: 'lines', positions, lineOffsets}`), `intersectionCapacity`,
`output`, optional `spatialSort` and `leafCapacity`. `output` is `{pieces, status?: {uncertainCount?}}`; `pieces` is a
`GPUGeneratedGeometryPort`: `geometry` holds `{kind: 'lines', positions, lineOffsets}`, `sourceIds`
identifies the caller's source feature (logical row without caller IDs), while optional
`sourcePartRows` retains the original input part row. `status` holds `count` (pieces, not vertices), `overflow`, optional `candidateOverflow`,
optional `requiredCount` and optional `vertexCount` and `requiredVertexCount`, which report the flattened vertex cardinality. Pieces are ordered by
source line, then traversal order. Splits at line ends are ignored, and a line with fewer than two
vertices yields no piece. Piece or vertex truncation sets `status.overflow`; an insufficient
`intersectionCapacity` or restricted BVH leaf coverage independently sets `status.candidateOverflow`.
For `GPULineSplit`, a supplied `leafCapacity` below `lines.positions.length` requires
`pieces.status.candidateOverflow`; final `pieces.status.overflow` alone cannot observe candidate loss.
Read both words before treating the pieces as complete. The final word also reports nonfinite input
or an uncertifiable intersection, which cannot be repaired by increasing output capacity.

`GPULineMerge` is the inverse (Shapely `linemerge`, turf `lineMerge`): it joins linestrings at endpoints
shared by exactly two line ends (an exact f32 match) into maximal chains. Junctions of three or more ends
stop a chain, a closed line never joins itself, and lines with fewer than two vertices are ignored. With
`directed: true` (Shapely `line_merge(directed=True)`) only a line end meeting a line start joins, chains
follow the input direction and `lineReversed` is all 0.
Props: `positions`, `lineOffsets` and `output: {chainOffsets, positions, status: {count, requiredCount?,
overflow?, vertexCount?, requiredVertexCount?}, sourceIds?, lineChains?, lineOrders?, lineReversed?}`;
`status.count` counts chains. The capacities always suffice, so `requiredCount` equals `count` and
`overflow` is 0. `sourceIds` is the source row of each chain, its first line; `lineChains` and
`lineOrders` list every member. Lines with fewer than two vertices receive `GPU_LINE_MERGE_NONE`
in `lineChains` and `lineOrders`, including after repeated submissions.
The zero overflow word is written on every submission, including when caller-owned status storage
previously held a nonzero value. It follows from the construction-time capacity requirements:
`positions` holds at least every input vertex and `chainOffsets` holds at least `lineCount + 1` rows.
Ignored lines receive zero in `lineReversed`, including when a previously reversed line becomes
a singleton on the next submission. `chainOffsets` may have more than `lineCount + 1` rows;
every row after the final chain repeats the written vertex total, including spare rows.
Merge captures view references, output descriptors, status fields and structural settings at
construction. Editing the caller's descriptor afterward does not replace validated destinations;
input buffer contents remain per-frame data and may be updated between submissions.
The list-ranking rounds are the shared helper in `line-merge/list-ranking.ts`, which `GPUSharedPaths` also uses. The result is deterministic: chains are ordered by their lowest line, paths start at the
lower endpoint ID and cycles start at their lowest line and are emitted closed. Chains are ranked by
pointer jumping (list ranking), so a chain of `n` lines takes `O(log n)` dispatches rather than `n`
serial steps. Merging with a tolerance is not provided.

```ts
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPULineSplit, GPULineMerge} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample3(graph: GPUCommandGraph, inputs: {
  positions: NonNullable<NonNullable<ConstructorParameters<typeof GPULineSplit>[0]['lines']>['positions']>;
  lineOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPULineSplit>[0]['lines']>['lineOffsets']>;
  piecePositions: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULineSplit>[0]['output']>['pieces']>['geometry']>['positions']>;
  offsets: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULineSplit>[0]['output']>['pieces']>['geometry']>['lineOffsets']>;
  lineIds: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULineSplit>[0]['output']>['pieces']>['sourceIds']>;
  count: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULineSplit>[0]['output']>['pieces']>['status']>['count']>;
  overflow: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULineSplit>[0]['output']>['pieces']>['status']>['overflow']>;
  vertexCount: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULineSplit>[0]['output']>['pieces']>['status']>['vertexCount']>;
  chainOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPULineMerge>[0]['output']>['chainOffsets']>;
  merged: NonNullable<NonNullable<ConstructorParameters<typeof GPULineMerge>[0]['output']>['positions']>;
}) {
  const {positions, lineOffsets, piecePositions, offsets, lineIds, count, overflow, vertexCount, chainOffsets, merged} = inputs;
  graph.add(new GPULineSplit({
    lines: {kind: 'lines', positions, lineOffsets}, intersectionCapacity: 1 << 16,
    output: {
      pieces: {
        geometry: {kind: 'lines', positions: piecePositions, lineOffsets: offsets},
        sourceIds: lineIds,
        status: {count, overflow, vertexCount}
      }
    }
  }));
  graph.add(new GPULineMerge({positions, lineOffsets, output: {chainOffsets, positions: merged, status: {count}}}));
}
```

`GPUNetworkNoding` in [GPU Network](/docs/api/gpu-network) builds a routable
network from the pieces.

#### How it computes

{/* lens:GPULineSplit */}

::lens{scenario=gpu-spatial-analysis.line-split rung=datum datum=0 caption="Piece 0: the first sub-line between intersections"}

::lens{scenario=gpu-spatial-analysis.line-split rung=kernel caption="The compute shader that writes the split pieces"}

::lens{scenario=gpu-spatial-analysis.line-split rung=graph caption="Command graph nodes and buffers that produce the split pieces"}

{/* lens:GPULineMerge */}

::lens{scenario=gpu-spatial-analysis.line-merge rung=datum datum=0 caption="Road 0: the merged chain it belongs to"}

::lens{scenario=gpu-spatial-analysis.line-merge rung=kernel caption="The compute shader that writes the merged chains"}

::lens{scenario=gpu-spatial-analysis.line-merge rung=graph caption="Command graph nodes and buffers that produce the merged chains"}

### `GPULineClipByPolygon` and `GPUSharedPaths`

**`GPULineClipByPolygon`** — Clips polylines against a polygon ring, keeping the pieces inside it.

::lens{scenario=gpu-spatial-analysis.line-clip-by-polygon rung=result caption="Grey: 223 Wicker Park streets · blue: pieces inside the West Town boundary"}

**`GPUSharedPaths`** — Finds the stretches that two sets of polylines share, writing them as runs with their source-line indices.

::lens{scenario=gpu-spatial-analysis.shared-paths rung=result caption="Grey: Chicago community-area boundaries · blue: stretches West Town shares with its neighbours"}

Maximum: each geometry has at most `getGPUSegmentIntersectionMaximumVertexCount(device)` (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) `positions` rows. `GPULineClipByPolygon` also needs `lines.positions` rows plus twice `intersectionCapacity` at most `getGPULineClipByPolygonMaximumSubPieceBound(device)` (4194304 on a default device); `GPUSharedPaths` needs `intersectionCapacity` at most `getGPUSharedPathsMaximumIntersectionCapacity(device)` (2796202 on a default device).

Both classes are planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first.

`GPULineClipByPolygon` clips linestrings to polygons (GeoPandas `clip(lines, polygons)`, Shapely
`intersection(line, polygon)`); `mode: 'outside'` is `difference`. It is bounded because it never computes
a new intersection between two polygons: every output vertex is an input vertex or a crossing of one line
segment with one polygon edge. `GPUSegmentIntersection` lists the crossings, each segment's crossings are
ranked in place (pairs arrive sorted by line segment) and deduplicated into sub-pieces. Sub-pieces inside a
collinear overlap are boundary and count as inside; the rest are classified by `GPUPointInPolygonJoin` at
their midpoint, and kept sub-pieces that continue each other merge. Several polygon features act as their
union. Props: `lines`, `polygons`, `mode`, `intersectionCapacity`, `candidateCapacity` and
`output: {pieces, status?: {uncertainCount}}` (`pieces` has the `GPULineSplitPieces` layout). Overflow of pairs,
candidates, pieces or vertices sets `output.pieces.status.overflow`. The optional
`output.pieces.status.candidateOverflow` identifies intersection-pair-list or BVH-leaf loss;
final piece/vertex capacity loss remains in the aggregate overflow word. Unlike split, clipping
folds candidate loss into aggregate overflow. A supplied `leafCapacity` below
`polygons.positions.length` requires `pieces.status.candidateOverflow` or `pieces.status.overflow`.
Both named piece status maps reject `complete` and `unresolvedCount` because neither has a writer. `status.count` counts pieces. Limits: a line is not noded at its own crossings, a line running over itself keeps both
traversals (Shapely dedupes), a point-only touch yields no piece (`keep_geom_type=True`), and a sub-piece a
few f32 ulps long beside an edge can be misclassified. Against Shapely 2.1.2, 52 lines in both modes
matched on total length (under 2e-3) and piece end points.

Uncertifiable intersections or midpoint classifications also set `pieces.status.overflow`,
whether or not `output.status.uncertainCount` is requested. An empty result with overflow set is
incomplete; increasing capacity alone cannot resolve predicate uncertainty. The optional count
reports the number of uncertifiable classifications for diagnosis.
Unused midpoint capacity is excluded from containment before candidate generation, so increasing
`intersectionCapacity` alone cannot introduce predicate uncertainty from padding rows.

Multipart `lines.featureOffsets` groups input parts without connecting them. Optional
`pieces.geometry.featureOffsets` maps logical features to their written pieces, including empty
and fully truncated features. `pieces.sourceIds` contains caller feature IDs (logical rows
without input `sourceIds`); optional `pieces.sourceFeatureIds` publishes the same identities.
`pieces.sourcePartRows` preserves original part rows, and `pieces.sourceFeatureRows` preserves
logical feature rows, independently of caller IDs.
`vertexSourceEdges` identifies the original outgoing edge at each written vertex; at a piece's
last vertex it identifies the incoming edge. Unwritten vertex rows use `0xffffffff`.

`GPUSharedPaths` is Shapely `shared_paths(a, b)` for many line pairs. Collinear overlaps from
`GPUSegmentIntersection` are oriented along the left line, tagged forward or backward by the exact sign of
the segment directions, and chained into one polyline per (left line, right line, direction). The output
`output: {runs, status}`: `runs` (`GPUSharedPathsRuns`) has `leftSourceIds` and `rightSourceIds` (the `left` and
`right` line rows of each run, contributor contract rule 8; **breaking (2026-11-10):** renamed from `leftLineIds` and
`rightLineIds`), `forward`, `offsets` and `positions`; `status` has `count`, `vertexCount?`, `overflow?`, `requiredCount?`, `requiredVertexCount?`,
`uncertainCount?` and `candidateOverflow?` (`count` counts runs, not vertices; `candidateOverflow` is 1 when the
`intersectionCapacity` pair list overflowed, which also sets `overflow`). Closed left lines are cut at their first vertex and two
coincident right segments emit duplicate runs. Runs are sized and written by list ranking (Wyllie pointer jumping, `ceil(log2(intersectionCapacity))`
rounds per pass) rather than one invocation walking each run, so a long shared road costs the same dispatches as a short one; the work is
`O(intersectionCapacity * log(intersectionCapacity))`.

`status.overflow` also reports uncertifiable intersection pairs, including when no run is written
and optional `uncertainCount` is absent. `candidateOverflow` reports intersection-pair-list or
restricted BVH-leaf loss. A zero run count with aggregate overflow set does not certify no overlap.
Output words must not overlap. `count`, `vertexCount`, `requiredCount`, `requiredVertexCount`,
`overflow` and `candidateOverflow` are co-bound in one scalar dispatch: their writable binding
ranges must remain disjoint after WebGPU storage-offset alignment. Adjacent packed words among
those co-bound columns are not a supported layout. `uncertainCount` is written by a separate
dispatch; distinct adjacent `overflow` and `uncertainCount` words are supported, as verified by
the shared-path packed-word regression. Other run columns also need disjoint aligned binding
ranges whenever co-bound.

The four run outputs (`leftSourceIds`, `rightSourceIds`, `forward`, `offsets`) and each side's
`...SourceIds`/`...PartRows` provenance pair require disjoint aligned bindings. A provenance
output cannot share a physical buffer with `status.count`, which that pass reads. Adjacent
`count`/`overflow` words at bytes 0 and 4 reject; adjacent `overflow`/`uncertainCount` remain
supported because they are written separately. With `leafCapacity < right.positions.length`, construction requires
`status.candidateOverflow` or aggregate `status.overflow`; aggregate overflow includes BVH and
intersection-list loss and predicate uncertainty. Candidate overflow diagnoses intermediate
capacity loss, while aggregate overflow additionally diagnoses final run/vertex capacity loss. Downstream completeness claims require aggregate
`overflow=0` evidence. An omitted aggregate view does not establish completeness, even when
`count` fits the run capacity or `requiredCount` is supplied.

Both operands accept multipart `featureOffsets`. Runs remain part-local, so a disconnected
feature never acquires a connecting edge. For both single-part and grouped operands, `leftSourceIds` and
`rightSourceIds` contain caller `sourceIds` (logical feature rows when absent); optional
`leftPartRows` and `rightPartRows` retain the original line-part rows. Empty logical features
emit no runs. Single-part inputs without caller IDs retain their row provenance. Supplied
single-part caller IDs were previously ignored; they are now honored.

#### How it computes

{/* lens:GPULineClipByPolygon */}

::lens{scenario=gpu-spatial-analysis.line-clip-by-polygon rung=datum datum=0 caption="Piece 0: the first street piece inside West Town"}

::lens{scenario=gpu-spatial-analysis.line-clip-by-polygon rung=kernel caption="The compute shader that writes the clipped pieces"}

::lens{scenario=gpu-spatial-analysis.line-clip-by-polygon rung=graph caption="Command graph nodes and buffers that produce the clipped pieces"}

{/* lens:GPUSharedPaths */}

::lens{scenario=gpu-spatial-analysis.shared-paths rung=datum datum=0 caption="Run 0: the first stretch West Town shares with a neighbour"}

::lens{scenario=gpu-spatial-analysis.shared-paths rung=kernel caption="The compute shader that writes the shared runs"}

::lens{scenario=gpu-spatial-analysis.shared-paths rung=graph caption="Command graph nodes and buffers that produce the shared runs"}

### `GPULineSimplification`

Simplifies polylines with Visvalingam or Douglas-Peucker importance, keeping vertices above a metre tolerance.

::lens{scenario=gpu-spatial-analysis.line-simplification rung=result caption="Grey: 223 Wicker Park streets · blue: simplified with a 5 m tolerance"}

Pass optional `spatialContext` to `getGPULineSimplificationParameterValues(settings)` to warn on short Douglas-Peucker `tolerance` values; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

`spatialContext` defaults to native planar coordinates, with the tolerance in position units. `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` (optional `sphereRadius`, default mean Earth radius) runs Douglas-Peucker (both metrics) with great-circle distances: `tolerance` and `importance` use the context's length unit (meters by default, now also kilometers with `units: 'kilometers'`), chords are straight in longitude/latitude, and the distance from a vertex to its chord is the haversine distance to the closest point of the chord (found in the local equirectangular frame of the vertex, exact to first order for chords of kilometers). The `'time-ratio'` expected position is interpolated linearly in longitude/latitude with the longitude difference wrapped. Great-circle distances use f32 transcendental functions, so the bit-exactness below holds for planar contexts only. `method: 'visvalingam'` is planar only and throws in a great-circle context (its tolerance is an area, its rounds mirror a bit-exact planar triangle area, and no spherical-excess variant exists); other metrics throw. For Visvalingam, or to keep planar bit-exactness, project first (`createGPULocalFrame`, `GPULocalFrameProjection`). Douglas-Peucker declares `measure: 'distance'` (a Lambert equal-area frame throws; use azimuthal-equidistant); Visvalingam declares `measure: 'area'` (an azimuthal-equidistant frame throws; use Lambert equal-area).

Douglas-Peucker (and approximate Visvalingam-Whyatt) simplification of many polylines or tracks, stored as rows sorted by line and
delimited by `lineOffsets`. It runs in two parts: a
per-vertex `importance` column computed once, and a per-frame selection at a tolerance from a
parameter buffer. Changing the tolerance re-runs only the mask, compaction, and publish nodes.

`lineOffsets` contains vertex indices into `positions` and may describe a subrange of its rows.
Equal adjacent offsets describe an empty line. A single in-bounds terminal describes zero lines
in the named operations. Computed importance and the selection mask are zero outside the
subrange. Selected IDs keep their indices in the full positions array, including when selection
reuses a supplied importance column. The GPU clamps offsets to the position row count. The CPU
forms accept `storage: 'resident-capacity'` for the same clamping. The resulting offsets must be
nondecreasing. With storage omitted, the CPU forms require nondecreasing, in-bounds offsets.
Set `storage: 'compact'` to require offsets that span all rows.

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPULineSimplification, getGPULineSimplificationParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample4(importanceGraph: GPUCommandGraph, frameGraph: GPUCommandGraph, inputs: {
  framePositions: GraphDataView<'float32x2'>;
  frameLineOffsets: GraphDataView<'uint32'>;
  frameImportance: GraphDataView<'float32'>;
  positions: NonNullable<ConstructorParameters<typeof GPULineSimplification>[0]['positions']>;
  lineOffsets: NonNullable<ConstructorParameters<typeof GPULineSimplification>[0]['lineOffsets']>;
  importance: NonNullable<ConstructorParameters<typeof GPULineSimplification>[0]['importance']>;
  converged: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULineSimplification>[0]['output']>['status']>['converged']>;
  iterationCount: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPULineSimplification>[0]['output']>['status']>['iterationCount']>;
  toleranceParameters: GPUParameterBuffer<'float32'>;
  ids: NonNullable<NonNullable<ConstructorParameters<typeof GPULineSimplification>[0]['output']>['ids']>;
  count: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  outputMask: GraphDataView<'uint32'>;
  lineCounts: GraphDataView<'uint32'>;
  lineStarts: GraphDataView<'uint32'>;
}) {
  const {framePositions, frameLineOffsets, frameImportance, positions, lineOffsets, importance, converged, iterationCount, toleranceParameters, ids, count, overflow, outputMask, lineCounts, lineStarts} = inputs;
  // One-shot graph: importance.
  importanceGraph.add(new GPULineSimplification({
    positions, lineOffsets, importance,
    metric: 'segment', // or 'time-ratio' (TD-TR) with float32 `timestamps`
    maximumIterations: 64,
    output: {status: {converged, iterationCount}}
  }));

  // Import the same caller-owned position/offset/importance buffers into frameGraph separately.
  // Per-frame graph: selection only.
  frameGraph.add(new GPULineSimplification({
    positions: framePositions, lineOffsets: frameLineOffsets, importance: frameImportance, computeImportance: false,
    parameters: toleranceParameters.importToGraph(frameGraph),
    output: {ids, status: {count, overflow}, outputMask, lineCounts, lineStarts}
  }));
  toleranceParameters.write(getGPULineSimplificationParameterValues({tolerance: 2})); // no recompile
}
```

- Importance: every line starts as one open interval between its endpoints (importance
  `+Infinity`). Each round, every open interval picks its farthest row (segmented argmax with ties
  to the smallest row) and splits there. The row's importance is
  `min(distance, min(importance[a], importance[b]))`, so importance never increases from a split
  to its children. An interval whose largest distance is 0 is resolved at once with importance 0.
- Rounds are GPU gated. They stop when no row is left (`status.converged = 1`) or after `maximumIterations`. The rounds are unrolled, so the command node count grows with `maximumIterations`: `estimatedCommandNodeCount` (exact for the importance rounds, excluding the `GPUCompaction` nodes of a row selection) is checked at construction against the shared 32,768-node budget (`validateCommandNodeBudget`); the largest accepted `maximumIterations` (1024) needs about five thousand nodes, so the budget is never the limit.
  The CPU twin also honors an explicitly supplied iteration limit and reports `converged = 0`
  when work remains, retaining the current importance. Unresolved Visvalingam rows have infinite
  importance; unresolved Douglas-Peucker rows retain their parent split bound. Omitting the CPU
  limit leaves its sequential computation uncapped; CPU Visvalingam counts sequential removals.
  A balanced split tree takes about `log2(n)` rounds and a spiral can take up to `n - 2`. If the
  cap is reached, the remaining rows get their interval's split importance, so the kept set is a
  superset of Douglas-Peucker's.
- Completion and selection capacity are independent. A two-endpoint line has no interior decisions:
  after one terminal round `converged = 1` and `iterationCount = 1`. With zero ID slots it still has
  `lineCounts = [2]` and `requiredCount = 2`, while `count = 0` and `overflow = 1`; with two slots,
  `count = 2` and `overflow = 0`. For an unfinished example, one Douglas-Peucker round on
  `(0,0),(1,1),(2,0),(3,1),(4,0)` chooses row 1 and leaves child interval `[1,4]` undecided when
  `finishSpanLimit = 1`, so a one-round budget gives `converged = 0`.
  `iterationCount` reports work actually performed, rather than the requested budget: on the
  four-point Visvalingam line `(0,0),(1,1),(2,1),(3,0)`, the first area-1/2 removal leaves one
  area-3/2 interior. Budget three therefore completes in two iterations with `converged = 1`.
- Selection: a row is kept when `importance > tolerance`. `output.ids` holds the kept rows in
  ascending order, and `status.count` is clamped so it can drive an indirect draw. `outputMask` is the
  per-row 0/1 mask. `lineCounts` (not
  clamped) and `lineStarts` give each line's range inside `output.ids`.
  Optional `output.status.candidateOverflow` is always zero: selection uses a full input-sized
  compaction buffer, so it has no intermediate candidate loss. Final `ids` truncation is reported
  by `output.status.overflow`.
  Both `GPULineSimplificationOperation` and `CPULineSimplificationOperation` accept the same
  `output.status.candidateOverflow` name when selecting IDs and publish a one-word zero. It
  requires `output.ids`; importance-only operations omit selection diagnostics. Diagnostic
  output buffers must be separate from inputs, including line offsets, so resubmission cannot
  overwrite the source layout. The same input-disjointness requirement applies to split's
  top-level `output.status.uncertainCount`.
  The contributor and both operation forms capture supported status fields by name, including
  inherited/non-enumerable getters, once at construction. Selection diagnostics `count`,
  `requiredCount`, `overflow` and `candidateOverflow` require `ids`; `converged` and
  `iterationCount` require importance computation. Unsupported status names such as
  `approximation` are refused. Settings, names and view destinations are immutable snapshots;
  later caller edits cannot change registered outputs, validated destinations or CPU capacity.
- Exactness: with `status.converged = 1`, the kept set at any tolerance `e` equals classic recursive
  Douglas-Peucker at `e` (split when `dmax > e`, first maximum on ties) under the same metric.
  The f32 distances are bit-identical to a `Math.fround` CPU oracle: products are prevented from
  fusing into FMAs, and division and square root are corrected to defined values.
- Metrics: `'segment'` measures the distance to the chord segment, which is the perpendicular
  distance inside the segment and the distance to the nearer anchor outside it (GEOS and
  simplify-js behave the same way). `'time-ratio'` is TD-TR's synchronized Euclidean distance,
  which needs float32 `timestamps`.
- Use tile- or view-local coordinates. Simplified lines may self-intersect, because topology is
  not preserved (Douglas-Peucker and Visvalingam alike).

**Visvalingam-Whyatt.** `method: 'visvalingam'` writes each vertex's effective triangle area (the context's
length unit squared; square meters in a default local frame) into `importance`, and the per-frame `tolerance` is then an area: a vertex is kept when
`importance > tolerance`, and endpoints are always kept. The selection outputs are the same as for
Douglas-Peucker. Each round removes every surviving vertex whose area is the smallest within
`neighborhoodRadius` steps both ways (1 to 8, default 3, compile-time), then unlinks them; the effective
area is the maximum of its own area and those of its removed neighbors, which is the sequential running
maximum. It is approximate: the sequential heap order cannot be parallelised exactly. Against
`simplification.simplify_coords_vw` on 3000-vertex random walks the kept set differed by 3.1, 0.65 and 0.13
percent of vertices at radius 1, 2 and 3, and lines up to 300 vertices matched exactly at radius 3. It needs
about 31 rounds for 300 vertices and 55 for 3000; `maximumIterations` (default 256) caps them, and undecided
rows get `+Infinity` (a superset). Not supported: `metric: 'time-ratio'` and topology-preserving VW
(`simplify_coords_vwp`).

#### How it computes

{/* lens:GPULineSimplification */}

::lens{scenario=gpu-spatial-analysis.line-simplification rung=datum datum=0 caption="Vertex 0: Douglas-Peucker importance in metres"}

::lens{scenario=gpu-spatial-analysis.line-simplification rung=kernel caption="The compute shader that writes the importance and retained vertices"}

::lens{scenario=gpu-spatial-analysis.line-simplification rung=graph caption="Command graph nodes and buffers that produce the importance and retained vertices"}

The CPU and GPU named line-simplification forms accept zero vertex rows, zero lines with one
terminal offset, and empty lines with repeated offsets. Importance and outputMask have vertex-row lengths;
lineCounts and lineStarts have line-row lengths, including zero. Selection ids may have capacity
zero. Scalar status views still have one word: count/requiredCount/overflow/candidateOverflow
are zero, while converged and iterationCount are one for empty work. The terminal input offset
is retained. The raw GPULineSimplification contributor retains its existing nonempty input contract.

{/* parameters: GPULineSimplification generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULineSimplification` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULineSimplification`.

- Packer: `getGPULineSimplificationParameterValues(settings)`, writes `GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPULineSimplificationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `tolerance` | `number` | Douglas-Peucker tolerance in position units (the context length unit in a great-circle context). A vertex is kept when its importance is strictly greater than the tolerance, which is the classic `dmax > epsilon` split rule. Line endpoints are always kept. Visvalingam uses an effective-area tolerance in squared position units. Must be finite and non-negative. |
| `method` (optional) | `GPULineSimplificationMethod` | Algorithm used by the operator. Defaults to `'douglas-peucker'`; `'visvalingam'` uses an area tolerance and skips the distance warning. Does not change the packed values. |

{/* /parameters */}

### `GPUCoverageSimplification`

An explicit `maximumIterations` must be an integer in `[1, 1024]`. The contributor validates this
budget at construction under its own ID, before building the nested line simplifier. Increase the
budget within that range when `output.status.converged` is zero; output capacity does not determine
solver convergence. The constructor reads and captures the budget once. Changing the caller's
`maximumIterations` afterward cannot change the built graph; construct a new contributor to use
a different budget.

Simplifies a polygon coverage with Douglas-Peucker, simplifying each shared border once so neighbours stay gap-free, and writes the kept vertices.

::lens{scenario=gpu-spatial-analysis.coverage-simplification rung=result caption="Grey: 17 community areas, 3,201 vertices · blue: 397 after 50 m simplification, borders still shared"}

Gap-free simplification of a polygon coverage (Shapely `simplify_coverage`, PostGIS `ST_CoverageSimplify`).
Every arc shared by two polygons is simplified once, so both neighbors keep the same vertices along it.
Arcs are runs of ring edges with the same partner polygon, broken at each ring start; the lower polygon ID
owns a shared arc. The kept vertices are the Douglas-Peucker survivors of the owned arcs plus the junctions,
shared through point IDs, and the output rings are the input rings filtered by the mask, in the same order
and direction.

Props: the four-level polygon layout (`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`; rows
are features and a ring's coverage identity is its feature),
`snapGridSize` (default 0, position units; zero keeps exact float32 coordinate identity).
Positive spacing must remain finite and positive after conversion to float32.
Cells use half-up rounding of `(coordinate - origin) / snapGridSize`, with negative half ties
rounding toward positive infinity, certified against the original operands even on reciprocal-division
backends. The origin is the first coverage vertex. Every vertex must be within
`2^24 * snapGridSize` of it on both axes; out-of-range keys increment optional
`status.uncertainCount`, while `overflow` retains its final vertex-capacity meaning.
At spacing 41, coordinate 20.5 maps to cell 1; coordinate 58720228 at spacing 7 stays in its
original cell. Cleanup uses a zero origin instead of this coverage origin.
Other props are `parameters` (packed with `getGPULineSimplificationParameterValues`; the
tolerance changes per frame without recompiling), `maximumIterations`, and `output` with
`positions` (the kept vertices, capacity), `ringOffsets` (`ringCount + 1`), optional `outputMask` (per input
vertex), and `status: {overflow, requiredCount?, converged?, uncertainCount?}`. Feature and polygon offsets are
unchanged.

Maximum: `positions` has at most `getGPUSegmentIntersectionMaximumVertexCount(device)` (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows.

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`). The tolerance is in position units. It measures distance, so `local-lambert-equal-area` frames throw.

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUCoverageSimplification, getGPULineSimplificationParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample5(graph: GPUCommandGraph, inputs: {
  positions: NonNullable<ConstructorParameters<typeof GPUCoverageSimplification>[0]['positions']>;
  featureOffsets: NonNullable<ConstructorParameters<typeof GPUCoverageSimplification>[0]['featureOffsets']>;
  polygonOffsets: NonNullable<ConstructorParameters<typeof GPUCoverageSimplification>[0]['polygonOffsets']>;
  ringOffsets: NonNullable<ConstructorParameters<typeof GPUCoverageSimplification>[0]['ringOffsets']>;
  toleranceBuffer: GPUParameterBuffer<'float32'>;
  outPositions: NonNullable<NonNullable<ConstructorParameters<typeof GPUCoverageSimplification>[0]['output']>['positions']>;
  outRingOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPUCoverageSimplification>[0]['output']>['ringOffsets']>;
  overflow: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUCoverageSimplification>[0]['output']>['status']>['overflow']>;
}) {
  const {positions, featureOffsets, polygonOffsets, ringOffsets, toleranceBuffer, outPositions, outRingOffsets, overflow} = inputs;
  graph.add(new GPUCoverageSimplification({
    positions, featureOffsets, polygonOffsets, ringOffsets,
    parameters: toleranceBuffer.importToGraph(graph),
    output: {positions: outPositions, ringOffsets: outRingOffsets, status: {overflow}}
  }));
  toleranceBuffer.write(getGPULineSimplificationParameterValues({tolerance: 0.05}));
}
```

Topology preservation: arc endpoints (junctions and ring starts) are always kept and a ring never keeps
fewer than three vertices (the farthest original vertices are restored, even with `topologyIterationCount: 0`).
Then `topologyIterationCount` (compile-time, default 4, 0 turns the repair off) iterations compact the
simplified rings, run `GPUSegmentIntersection` with exact predicates and, for every pair of segments that
cross, touch or overlap without sharing a coverage point, restore the original vertex farthest from each
segment (one Douglas-Peucker split; decisions go through point IDs, so neighbors still agree). Each
iteration also tests every kept vertex against the area between each span's original chain and its shortcut
(exact point-in-ring signs, candidates from one BVH over the input segments): a shortcut that passes over an
island crosses nothing but would move the island into the neighboring polygon, so such a span is repaired
the same way. `output.topologyStatistics` (4 `uint32`, length
`GPU_COVERAGE_SIMPLIFICATION_TOPOLOGY_STATISTICS_LENGTH`) reports the crossings and swept spans before repair,
those remaining, the vertices restored and a candidate-pair overflow flag; `topologyPairCapacity` (default `max(256, 4 * vertices)`) sizes the candidate list.
If any topology candidate-pair detection overflows, every original coverage vertex is restored,
including when `output.topologyStatistics` is omitted. This conservative fallback is independent
of statistics selection. Public `status.overflow` continues to mean final vertex-capacity loss:
the adjacent-polygon fixture requires 10 original vertices at pair capacity 1 versus 8 at 256. Residual crossings
are reported, never hidden, and input that already crosses itself cannot be repaired (it counts in both
numbers). On a wavy-strip scene the repaired output passes Shapely `coverage_is_valid`, where plain
Douglas-Peucker gave three invalid polygons. The criterion is distance, not area (Shapely's
`simplify_coverage` is area based and keeps a different vertex set), and there is no guarantee of zero
residual within the iteration count.

`simplifyBoundary: false` (Shapely `coverage_simplify(simplify_boundary=False)`) keeps every vertex that
touches an unshared edge and simplifies only the shared arcs.

One thread walks each arc, so one huge arc is a serial walk (and a long dropped span costs a serial walk
per kept vertex), and non-finite vertices are not supported. A shared arc that crosses a ring start keeps
one extra vertex.

#### How it computes

{/* lens:GPUCoverageSimplification */}

::lens{scenario=gpu-spatial-analysis.coverage-simplification rung=datum datum=0 caption="Vertex 0: a ring start, always kept as an arc endpoint"}

::lens{scenario=gpu-spatial-analysis.coverage-simplification rung=kernel caption="The compute shaders that mark arcs, simplify them and repair crossings"}

::lens{scenario=gpu-spatial-analysis.coverage-simplification rung=graph caption="Command graph nodes and buffers that produce the kept vertices"}

{/* parameters: GPUCoverageSimplification generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUCoverageSimplification` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUCoverageSimplification`.

- Packer: `getGPULineSimplificationParameterValues(settings)`, writes `GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH` (4) `float32` values.
  - Input port: `parameters`.

Settings of `getGPULineSimplificationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `tolerance` | `number` | Douglas-Peucker tolerance in position units (the context length unit in a great-circle context). A vertex is kept when its importance is strictly greater than the tolerance, which is the classic `dmax > epsilon` split rule. Line endpoints are always kept. Visvalingam uses an effective-area tolerance in squared position units. Must be finite and non-negative. |
| `method` (optional) | `GPULineSimplificationMethod` | Algorithm used by the operator. Defaults to `'douglas-peucker'`; `'visvalingam'` uses an area tolerance and skips the distance warning. Does not change the packed values. |

{/* /parameters */}

### `GPUCoverageDissolve`

Merges the features of a polygon coverage that share a label, writing the boundary rings of each merged region.

::lens{scenario=gpu-spatial-analysis.coverage-dissolve rung=result caption="Grey: 17 community areas · blue: 4 rings after dissolving three west-to-east bands (labels from easting)"}

Dissolves a polygon coverage by a per-feature `uint32` label (the four-level polygon layout; GeoPandas `dissolve(by, method='coverage')`,
Shapely `coverage_union_all` per label). An edge is dropped when another polygon with the same label has
the same edge (snapped point IDs, `(min, max)` keys). Kept edges are oriented interior-left and chained by
`GPUSegmentRingAssembly` with the label as the group, so the output has shells, holes and
`polygons.sourceIds`. Vertices at nodes are kept, as GEOS does. This is not the excluded vector
overlay: a coverage has no crossings, so the result is a subset of the input edges and is bounded by the
input vertex count plus one closing vertex per ring, with ring and vertex capacities and an overflow flag.
The input must be a valid coverage (check it with [`GPUCoverageValidity`](./operators-validity.md#gpucoveragevalidity)).
Planar only: `spatialContext` defaults to native planar coordinates and `longitude-latitude` throws.
`snapGridSize` (position units, default 0) snaps vertices relative to the first vertex of
`positions`, using the original-operand-certified half-up rule above. Positive spacing must
survive finite positive float32 conversion. Every vertex must be within `2^24 * snapGridSize`
of the origin on both axes; out-of-range keys increment optional `status.uncertainCount`, and
`overflow` retains its final capacity meaning. `status.complete` includes snap-key uncertainty
even when `status.uncertainCount` is omitted. Optional input `labelsComplete` is separate,
dimensionless solver provenance (1 complete, 0 provisional), and requires
`output.status.labelsComplete`. The output word is copied every submission, defaulting to 1
without solver input. Provisional labels keep useful incumbent geometry and do not change
vertex/ring overflow or geometric/snap completeness. This distinction also applies when
labels are categorical and the optional input is omitted. After keying,
the output uses the coordinates of the lowest vertex index of each point. Ring winding is the exact turn at
each ring's lexicographically smallest vertex. Dissolve edge sets,
area, parts and holes matched `gdf.dissolve(method='coverage')` exactly on 5 scenes and 18 labels. The
per-ring area walk is serial per ring. The directory is `polygon-coverage-topology/` because `.gitignore`
ignores `coverage-*/`.

#### How it computes

{/* lens:GPUCoverageDissolve */}

::lens{scenario=gpu-spatial-analysis.coverage-dissolve rung=datum datum=0 caption="Area 0 (community area 3): its band label"}

::lens{scenario=gpu-spatial-analysis.coverage-dissolve rung=kernel caption="The compute shaders that match edges and orient the surviving boundary"}

::lens{scenario=gpu-spatial-analysis.coverage-dissolve rung=graph caption="Command graph nodes, sorts and ring assembly that produce the dissolved rings"}

### `GPUTrajectoryMetrics`

Computes per-track length, duration, average and maximum speed, and a bounded stop list from time-ordered positions.

::lens{scenario=gpu-spatial-analysis.trajectory-metrics rung=result caption="Grey: 12 AIS tracks · right: the same tracks coloured by average speed in m/s"}

Per-track metrics for rows sorted by track and time and delimited by `trackOffsets`: path length,
duration, average speed, and maximum speed (via `GPUSegmentedReduction`). Stop detection finds
maximal runs of slow steps (speed below a per-frame threshold) that last at least a per-frame
minimum duration. It publishes a bounded stop list in row order (track, start row, end row,
duration, centroid) and unclamped per-track stop counts.

`timestamps` may be `float32`, `float32` plus `timestampsLow` (double-single), or exact `Int64`
words (`uint32x2`). Every time difference (steps, track durations, dwells) is taken exactly or in
double-single before conversion to float32, so multi-year epoch-millisecond tracks keep
millisecond steps. Durations and `stopMinimumDuration` use the timestamp unit; speeds and
`stopSpeedThreshold` use the context's length unit per timestamp unit.

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUTrajectoryMetrics, getGPUTrajectoryMetricsParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample6(graph: GPUCommandGraph, inputs: {
  positions: NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['positions']>;
  timestamps: NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['timestamps']>;
  trackOffsets: NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['trackOffsets']>;
  stopParameters: GPUParameterBuffer<'float32'>;
  trackLengths: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['output']>['trackLengths']>;
  averageSpeeds: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['output']>['averageSpeeds']>;
  maximumSpeeds: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['output']>['maximumSpeeds']>;
  trackStopCounts: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['output']>['trackStopCounts']>;
  stopTracks: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['output']>['stops']>['ids']>;
  count: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['output']>['stops']>['status']>['count']>;
  overflow: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['output']>['stops']>['status']>['overflow']>;
  startRows: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['output']>['stops']>['startRows']>;
  endRows: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['output']>['stops']>['endRows']>;
  centroids: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['output']>['stops']>['centroids']>;
  durations: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryMetrics>[0]['output']>['stops']>['durations']>;
}) {
  const {positions, timestamps, trackOffsets, stopParameters, trackLengths, averageSpeeds, maximumSpeeds, trackStopCounts, stopTracks, count, overflow, startRows, endRows, centroids, durations} = inputs;
  graph.add(new GPUTrajectoryMetrics({
    positions, timestamps, trackOffsets, parameters: stopParameters.importToGraph(graph),
    output: {
      trackLengths, averageSpeeds, maximumSpeeds, trackStopCounts,
      stops: {ids: stopTracks, status: {count, overflow}, startRows, endRows, centroids, durations}
    }
  }));
  stopParameters.write(getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 0.5, stopMinimumDuration: 300}));
}
```

Optional per-vertex columns (`positions.length` float32 rows) describe the step that ends at each row:
`stepSpeeds` (distance over delta time, 0 if the delta time is not positive), `stepHeadings` (`atan2(dy,
dx)` in radians, 0 for a zero-distance step) and `stepAccelerations` (`(speed_i - speed_{i-1}) /
deltaTime_i`, 0 for the first two rows of a track). The first row of a track and rows outside tracks are 0,
so a drawn segment takes the value of its end row. The track inputs are the trajectory layout
(`positions`, `timestamps`, `trackOffsets`; `GPUTrajectoryGeometryFields`, timestamps `float32` or Int64
`uint32x2` words), every result view goes in the
`output` bag, and `output.stops` is a `GPUCompactOutput` extended with the per-stop columns. The packer
takes a `GPUTrajectoryMetricsSettings`. `output.stops.drawInstanceCount` (one `uint32` row) receives
the clamped stop count, for example an indirect draw record's instance count.

`spatialContext` is optional. Omitted (or planar) means Euclidean distances in position units and headings
`atan2(dy, dx)`. With `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` (optional
`sphereRadius`, default the mean Earth radius in the context's length unit) the positions are `(longitude, latitude)` degrees,
step distances and `trackLengths` are in the context's length unit (meters by default; kilometers with
`units: 'kilometers'`); speeds and `stopSpeedThreshold` use that unit per timestamp unit.
`stepHeadings` are initial bearings in the planar heading convention, radians counterclockwise from
east in `(-pi, pi]` (`pi / 2 - bearing`, where `GPUGeodesicPairs` reports the bearing in degrees clockwise
from north), so headings mean the same in every context. Longitude/latitude without the `great-circle` metric throws. It measures distance, so `local-lambert-equal-area` frames throw.
Stop `centroids` are arithmetic means of coordinates, so a dwell straddling the antimeridian or a pole has a
meaningless centroid. For stop detection, the affected distance is `stopSpeedThreshold` × sampling
interval: float32 position noise is about ±0.8 m/s at 1 Hz in Chicago, so use a local frame.
No warning is possible because the distance is a GPU view; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).
The `spatialContext` prop used to be required and
planar-only.

#### How it computes

{/* lens:GPUTrajectoryMetrics */}

::lens{scenario=gpu-spatial-analysis.trajectory-metrics rung=datum datum=0 caption="Track 0 (MMSI 366993970): length, duration, average speed and stops"}

::lens{scenario=gpu-spatial-analysis.trajectory-metrics rung=formula caption="Track length as summed step distances; average speed is length over duration"}

::lens{scenario=gpu-spatial-analysis.trajectory-metrics rung=kernel caption="The compute shaders that write the track metrics and stops"}

::lens{scenario=gpu-spatial-analysis.trajectory-metrics rung=graph caption="Command graph nodes and buffers that produce the track metrics"}

{/* parameters: GPUTrajectoryMetrics generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTrajectoryMetrics` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUTrajectoryMetrics`.

- Packer: `getGPUTrajectoryMetricsParameterValues(settings)`, writes `GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUTrajectoryMetricsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `stopSpeedThreshold` | `number` | Speed threshold in the context's length unit per timestamp unit. A step is slow below this value. Must be finite and non-negative. A threshold of 0 only treats zero-distance steps as slow. |
| `stopMinimumDuration` | `number` | A dwell (a maximal run of slow steps) is reported as a stop when its duration is at least this value. Must be finite and non-negative. |

{/* /parameters */}

### `GPUTrajectoryPlayhead` and `GPUTrajectoryResample`

**`GPUTrajectoryPlayhead`** — Interpolates every track at a playhead time, writing position, heading, speed and status per track.

::lens{scenario=gpu-spatial-analysis.trajectory-playhead rung=result caption="Grey: 12 AIS tracks · dots: each track at 15:00Z by status, 9 active"}

**`GPUTrajectoryResample`** — Resamples each track to a fixed number of points, spaced evenly in time, path length or on a shared clock.

::lens{scenario=gpu-spatial-analysis.trajectory-resample rung=result caption="Grey: 12 AIS tracks · blue: 16 samples per track, evenly spaced along path length"}

`GPUTrajectoryPlayhead` interpolates every track at a per-frame playhead: one invocation per track
binary-searches the track's sorted timestamps and interpolates the bracketing segment. It writes
per-track columns (position, optional elevation, heading in radians, speed, status, segment row and
fraction) and an optional compact, ascending list of active tracks with a clamped count, so a
layer can draw one instance per moving object with an indirect instance count. Moving the playhead
only rewrites the parameter buffer.

Maximum (`GPUTrajectoryPlayhead`): the track count is at most `getGPUTrajectoryPlayheadMaximumTrackCount(device)` (6710886 on a default device).

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUTrajectoryPlayhead, getGPUTrajectoryPlayheadParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample7(graph: GPUCommandGraph, inputs: {
  positions: NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['positions']>;
  timestamps: NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['timestamps']>;
  trackOffsets: NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['trackOffsets']>;
  elevations: NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['elevations']>;
  playheadParameters: GPUParameterBuffer<'float32'>;
  currentPositions: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['output']>['currentPositions']>;
  currentElevations: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['output']>['currentElevations']>;
  headings: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['output']>['headings']>;
  speeds: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['output']>['speeds']>;
  trackStatus: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['output']>['trackStatus']>;
  segmentRows: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['output']>['segmentRows']>;
  segmentFractions: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['output']>['segmentFractions']>;
  ids: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['output']>['activeTracks']>['ids']>;
  count: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['output']>['activeTracks']>['status']>['count']>;
  overflow: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['output']>['activeTracks']>['status']>['overflow']>;
  drawInstanceCount: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryPlayhead>[0]['output']>['drawInstanceCount']>;
}) {
  const {positions, timestamps, trackOffsets, elevations, playheadParameters, currentPositions, currentElevations, headings, speeds, trackStatus, segmentRows, segmentFractions, ids, count, overflow, drawInstanceCount} = inputs;
  graph.add(new GPUTrajectoryPlayhead({
    positions, timestamps, trackOffsets, // same layout as GPUTrajectoryMetrics
    elevations, // optional float32 z per row
    parameters: playheadParameters.importToGraph(graph),
    output: {
      currentPositions, currentElevations, headings, speeds, trackStatus, segmentRows, segmentFractions,
      activeTracks: {ids, status: {count, overflow}}, drawInstanceCount
    }
  }));
  playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead: 125.5, maximumGap: 60}));
  // Int64 epoch-ms words: a uint32 parameter buffer and
  // getGPUTrajectoryPlayheadWordParameterValues({playhead: Date.now() + 0.25, maximumGap: 60_000})
}
```

- Status (`output.trackStatus`, `GPU_TRAJECTORY_PLAYHEAD_STATUS`): `empty` (no rows), `active`, `beforeStart`,
  `afterEnd`, and `gap` (the playhead lies strictly inside an interval longer than `maximumGap`; 0
  disables the rule). Outside the range the position clamps to the first or last sample and the
  heading is that of the first or last segment; in a gap the position holds at the last fix.
  Speed is non-zero only for active tracks.
- The segment is `[j - 1, j]` with `t[j - 1] <= p < t[j]`, or the last segment when `p` equals the
  last time. A playhead exactly on a sample time returns that sample's position bit-exactly.
- Duplicate timestamps: the last row of a run of equal times wins ("latest row wins"), so the
  interpolation interval always has a positive duration and nothing divides by zero.
- Heading is `atan2(dy, dx)` (0 along +x, counterclockwise); compass bearing is `π/2 - heading`.
  Zero-length segments give 0. Positions are planar unless `spatialContext` is great-circle.
- Great-circle (`spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`, optional
  `sphereRadius`, default the mean Earth radius in the context's length unit): positions are `(longitude, latitude)` degrees and
  each segment is interpolated by spherical linear interpolation instead of a linear blend of degrees. The
  heading is the segment's initial bearing in the planar convention (radians counterclockwise from east,
  `pi / 2 - bearing`, in `(-pi, pi]`), and `speeds` use the context's length unit per timestamp unit
  (meters by default; kilometers with `units: 'kilometers'`). Interpolated longitudes are continuous with the segment's start
  row, so a segment that crosses the antimeridian yields longitudes beyond ±180 (the end row is `b` shifted
  by a multiple of 360 degrees) rather than jumping; wrap downstream if needed. Fractions 0 and 1 return
  the start and end row exactly (up to that shift), including antipodal endpoints. Interior
  interpolation within 1e-6 radians of an antipode returns NaN because that path is not certified. Other contexts throw. Ellipsoidal interpolation is a
  non-goal. Both playhead and resample measure distance, so `local-lambert-equal-area` frames throw.
- Time: comparisons are exact in both modes (Int64 words compare integer words, then the f32
  playhead fraction); differences are subtracted exactly and then rounded to f32.

`GPUTrajectoryResample` resamples every track to `sampleCount` samples, uniformly in time
(`spacing: 'time'`, default) or along the path (`'arc-length'`: length in the context's unit; a
great-circle `spatialContext` also slerps the samples as the playhead does), into a dense
`[trackCount × sampleCount]` `float32x2` buffer (row `track * sampleCount + k`), with optional
elevations and sample times relative to the track start. It is the fixed-length input for
trajectory similarity, k-means, and fixed-vertex trails.

```ts
import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUTrajectoryResample} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function declareTrajectoryResample(device: Device) {
  const graph = new GPUCommandGraph(device);
  const usage = Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC;
  const positionsBuffer = device.createBuffer({data: Float32Array.of(0, 0, 1, 0, 2, 0), usage});
  const timestampsBuffer = device.createBuffer({data: Float32Array.of(0, 1, 2), usage});
  const offsetsBuffer = device.createBuffer({data: Uint32Array.of(0, 3), usage});
  const elevationsBuffer = device.createBuffer({data: Float32Array.of(0, 10, 20), usage});
  const positions = graph.createDataView(graph.importBuffer({id: 'positions', byteLength: 24, usage}, positionsBuffer), {format: 'float32x2', length: 3});
  const timestamps = graph.createDataView(graph.importBuffer({id: 'timestamps', byteLength: 12, usage}, timestampsBuffer), {format: 'float32', length: 3});
  const trackOffsets = graph.createDataView(graph.importBuffer({id: 'tracks', byteLength: 8, usage}, offsetsBuffer), {format: 'uint32', length: 2});
  const elevations = graph.createDataView(graph.importBuffer({id: 'elevations', byteLength: 12, usage}, elevationsBuffer), {format: 'float32', length: 3});
  // One initialized three-fix track produces 32 samples.
  const samples = createTransientView(graph, 'samples', 'float32x2', 32);
  const sampleElevations = createTransientView(graph, 'sample-elevations', 'float32', 32);
  const sampleTimes = createTransientView(graph, 'sample-times', 'float32', 32);
  graph.add(new GPUTrajectoryResample({positions, timestamps, trackOffsets, elevations,
    sampleCount: 32, spacing: 'arc-length', output: {samples, sampleElevations, sampleTimes}}));
  return {graph, samples, sampleElevations, sampleTimes, inputBuffers: [positionsBuffer, timestampsBuffer, offsetsBuffer, elevationsBuffer]};
}
```

Both take the trajectory layout (`positions`, `timestamps`, `trackOffsets`) as flat props, and all result
views in `output`. The playhead packer takes a `GPUTrajectoryPlayheadSettings` and the clock packers a
`GPUTrajectoryClockSettings`.

`spacing: 'clock'` samples every track at the instants of one shared clock (a per-frame `clock` view
`[start, step]`, packed with `getGPUTrajectoryClockParameterValues` or, for Int64 timestamps,
`getGPUTrajectoryClockWordParameterValues`; `GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH` words). Column `k` is the
same instant for every track and the sample is NaN outside a track's own time span, which is the layout
`GPUTrajectoryEncounters` expects. With `spacing: 'clock'` or `'time'`, a sample is NaN too when it lies
strictly inside a bracketing interval longer than the optional per-frame `maximumGap` view (one `float32`
row, the timestamps' unit); see the gap rule below. Omitted or 0 means no limit, the previous behaviour.

- Sample `k` targets `total * k / (sampleCount - 1)`; the last sample is the last row exactly. It
  uses the playhead's search and duplicate rule. Single-row tracks repeat that row; empty tracks
  write zeros.
- Arc length sums step lengths sequentially per track (deterministic, O(track length) per
  invocation).
- Inputs must be single packed views; chunked vectors are not supported.

#### How it computes

{/* lens:GPUTrajectoryPlayhead */}

::lens{scenario=gpu-spatial-analysis.trajectory-playhead rung=datum datum=1 caption="Track 1 (MMSI 366999411): bracketing segment, position and status at 3600 s"}

::lens{scenario=gpu-spatial-analysis.trajectory-playhead rung=formula caption="Interpolated position at the playhead; speed is segment length over duration"}

::lens{scenario=gpu-spatial-analysis.trajectory-playhead rung=kernel caption="The compute shader that searches the bracketing segment and interpolates"}

::lens{scenario=gpu-spatial-analysis.trajectory-playhead rung=graph caption="Command graph nodes and buffers that produce the playhead columns"}

{/* lens:GPUTrajectoryResample */}

::lens{scenario=gpu-spatial-analysis.trajectory-resample rung=datum datum=0 caption="Sample 0: the first point of track 0 and its time since the track start"}

::lens{scenario=gpu-spatial-analysis.trajectory-resample rung=formula caption="Target progress per sample, then interpolation inside the bracketing segment"}

::lens{scenario=gpu-spatial-analysis.trajectory-resample rung=kernel caption="The compute shader that writes the resampled points"}

::lens{scenario=gpu-spatial-analysis.trajectory-resample rung=graph caption="Command graph nodes and buffers that produce the resampled points"}

#### The `maximumGap` rule

`GPUTrajectoryPlayhead`, `GPUTrajectoryResample` with clock or time spacing, `addClockEncounters`,
`GPUTrajectoryClosestApproach`, `GPUZoneEvents`, `GPUTrajectoryKalman` and `GPUTrajectoryRestriction`
use one rule. `maximumGap` is in the unit of the timestamps (word timestamps: the integer unit, and
`context.unit` when a `temporalContext` is declared). A step between two consecutive samples is a gap only
when it is strictly longer than the limit; a step exactly equal to the limit is still interpolated.
The samples at both ends of a gap are real fixes and stay present, so only times strictly inside the
gap are affected. Omitted (views) or 0 (everywhere) means no limit. Arc-length resampling takes no
limit: its samples are spaced along the path, and a straight line across a gap is still path length.

#### Trajectory port and sort check

These contributors share the `GPUTrajectoryGeometryPort` layout, including Kalman and restriction,
which also keep their `times` and `offsets` names. The optional `temporalContext` must match the rows:
`float32` is `float32-relative` (`double-single` with `timestampsLow`), `uint32x2` is `int64-words`,
and the axis is `{kind: 'irregular'}`; a mismatch throws at construction.

`addTrajectorySortCheck(graph, {positions, timestamps, trackOffsets, output: {status: {invalidCount}}})`
is an optional check of the sorted-by-track-then-time precondition, which nothing else validates. It
writes one `uint32` row, the number of rows out of order, and costs nothing unless called.
Each non-finite float timestamp counts once, including a track's first or only row. For
double-single timestamps, either non-finite component makes the row invalid. A finite row after
an invalid row starts a new ordered run and is not counted again for that predecessor. Int64
timestamps retain exact word comparisons. A reversed offset range counts the rows in that range.
`status.invalidCount` must contain at least one row; an empty view throws before any node is added.

{/* parameters: GPUTrajectoryPlayhead generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTrajectoryPlayhead` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUTrajectoryPlayhead`.

- Packer: `getGPUTrajectoryPlayheadParameterValues(playhead)`, writes `GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH` (4) `float32` values.
- Packer: `getGPUTrajectoryPlayheadWordParameterValues(playhead)`, writes `GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH` (4) `uint32` values.

Settings of `getGPUTrajectoryPlayheadParameterValues`:

| Field | Type | Description |
|---|---|---|
| `playhead` | `number \| bigint` | Playhead instant. Float32 mode: a number in the timestamps' relative epoch. Word mode: a `bigint` (exact integer) or a `number` whose fractional part becomes the f32 sub-unit fraction. |
| `maximumGap` (optional) | `number` | Largest bracketing interval that is still interpolated, in time units. When the playhead lies strictly inside a longer interval the track status is `gap`. 0 (default) disables the rule. |

{/* /parameters */}

{/* parameters: GPUTrajectoryResample generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTrajectoryResample` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUTrajectoryResample`.

- Packer: `getGPUTrajectoryClockParameterValues(clock)`, writes `GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH` (4) `float32` values.
  - Input port: `clock`.
- Packer: `getGPUTrajectoryClockWordParameterValues(clock)`, writes `GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH` (4) `uint32` values.
  - Input port: `clock`.

Settings of `getGPUTrajectoryClockParameterValues`:

| Field | Type | Description |
|---|---|---|
| `start` | `number \| bigint` | First clock instant. Float32 mode: a number in the timestamps' relative epoch. Word mode: a `bigint` (exact integer) or a `number` whose fractional part becomes the f32 sub-unit fraction. |
| `step` | `number` | Distance between consecutive clock samples, finite and positive. |

{/* /parameters */}

### `GPUZoneEvents`

Candidate capacities reserve a strictly higher saturated-count sentinel; device bindings usually
impose much lower limits.

**`GPUZoneEvents`** — Finds when each track enters and exits polygon zones, with crossing times and dwell time and visits per track and zone.

::lens{scenario=gpu-spatial-analysis.zone-events rung=result caption="Right: 12 AIS tracks coloured by zone entries and exits, 111 events in all · grey: zones"}

**`addFleetDwellZoneEventsRecipe`** — Chains zone events and a per-zone roll-up into total dwell time and visiting-track counts per zone.

::lens{scenario=gpu-spatial-analysis.fleet-dwell-zone-events-recipe rung=result caption="Right: 13 harbour zones coloured by total vessel dwell in seconds · grey: tracks"}

With the optional per-frame `maximumGap` view (one `float32` row, the timestamps' unit), a segment
longer than the limit is a data gap. Crossings on it are reported at the first fix after the gap
(its time, position and row) and the gap adds no dwell, even when the track is inside the zone on
both sides of it. Omitted or 0 keeps the previous straight-line behaviour.

Entry and exit events of trajectories against polygon zones, with interpolated times, bounded events per
track, and dwell time and visit count per `(track, zone)`.

`spatialContext` is planar only (the default). Segments and zone edges are straight lines in the coordinate
plane, which is not the geometry of degrees (a great-circle segment is not straight in longitude/latitude, and
edges cross the antimeridian), so any longitude-latitude context throws; project positions and zones first
with `createGPULocalFrame`.

- Inputs: `tracks`, a tagged trajectory geometry (`kind: 'trajectories'`, `positions` `float32x2`,
  `timestamps` `float32` relative or `uint32x2` Int64 words, `trackOffsets` with `trackCount + 1` rows), and
  `zones`, a tagged polygon geometry (`kind: 'polygons'`) in the four-level layout (`positions`,
  `featureOffsets`, `polygonOffsets`, `ringOffsets`). Zone `z` is feature `z`, so the zone count is
  `zones.featureOffsets.length - 1`; rings of one feature (holes, extra parts) share the zone index and the
  first ring of a polygon is its shell. Rings close implicitly, a repeated closing vertex is tolerated, and
  rings with fewer than three vertices or non-finite vertices are ignored. A first kernel derives one edge per
  zone vertex from the layout. Plus `candidateCapacity` and `trackEventCapacity` (events kept per track).
- Outputs, all in `output`: `events` (a `GPUCompactOutput` extended with the columns: `ids` is the track index),
  optional `eventZones`, `eventTypes` (`GPU_ZONE_EVENT_TYPE.enter` 0, `.exit` 1), `eventTimes` (relative to
  the track's first timestamp, f32) and `eventRows` (the segment end row); optional dense
  `[trackCount * zoneCount]` `dwellTimes` and `visitCounts`; and `trackEventCounts` (unclamped).
- Events are sorted by `(track, time, edge row)`. Each segment queries a `GPUBVH` of edge bounds and
  runs an exact crossing test (`orientSign`, never f32 parameters); the time is
  `t[r - 1] + s * (t[r] - t[r - 1])`. Enter versus exit comes from parity, not ring orientation: the first
  sample of every track probes with a +x ray (even-odd), so a track that starts inside a zone works, and
  that first interval counts as a visit with dwell from `t = 0`. Open visits close at the track's last
  sample. Dwell and visits use every event, even when the event list is truncated. No kernel walks a
  track's events one by one: a stable sort by `(track, zone)` makes the events of each cell a
  time-ordered run, so event types, ranks, visit counts and spans are O(1) per event or cell and dwell
  is a load-balanced segmented sum. A track with thousands of crossings costs no more than many short
  tracks with the same event total.
- Boundary rule (the library's shared Simulation-of-Simplicity rule, `utils/exact-geometry-wgsl.ts`): every track sample `(x, y)` is read as `(x - e^2, y - e)` for an infinitesimal `e > 0`
  (the track moves, the polygon does not), for the initial ray and the segment crossings alike, so the
  parity of a track's crossings always equals the change of its inside flag. A track that only touches a
  polygon apex or an edge from outside has no event; one that runs through a vertex into the interior has
  one; a sample on an edge is just below it (inside when the interior lies below) and on a vertical edge
  just west of it. Event times along a collinear overlap use the f32 projection.
- `output.events.status.overflow` is 1 when more events were kept than `ids.length`. Optional
  `output.events.status.candidateOverflow` reports candidate scratch or per-track event truncation.
  `requiredCount` counts retained events before final truncation, so it is a lower bound when
  candidate scratch or `trackEventCapacity` dropped events.

```ts
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUZoneEvents} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample9(graph: GPUCommandGraph, inputs: {
  positions: NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['tracks']>['positions']>;
  timestamps: NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['tracks']>['timestamps']>;
  trackOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['tracks']>['trackOffsets']>;
  zonePositions: NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['zones']>['positions']>;
  featureOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['zones']>['featureOffsets']>;
  polygonOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['zones']>['polygonOffsets']>;
  ringOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['zones']>['ringOffsets']>;
  ids: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['output']>['events']>['ids']>;
  count: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['output']>['events']>['status']>['count']>;
  overflow: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['output']>['events']>['status']>['overflow']>;
  eventCandidateOverflow: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['output']>['events']>['status']>['candidateOverflow']>;
  eventZones: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['output']>['events']>['eventZones']>;
  eventTypes: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['output']>['events']>['eventTypes']>;
  eventTimes: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['output']>['events']>['eventTimes']>;
  dwellTimes: NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['output']>['dwellTimes']>;
  visitCounts: NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['output']>['visitCounts']>;
  candidateScratchOverflow: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUZoneEvents>[0]['output']>['status']>['candidateOverflow']>;
}) {
  const {positions, timestamps, trackOffsets, zonePositions, featureOffsets, polygonOffsets, ringOffsets, ids, count, overflow, eventCandidateOverflow, eventZones, eventTypes, eventTimes, dwellTimes, visitCounts, candidateScratchOverflow} = inputs;
  graph.add(new GPUZoneEvents({
    tracks: {kind: 'trajectories', positions, timestamps, trackOffsets},
    zones: {kind: 'polygons', positions: zonePositions, featureOffsets, polygonOffsets, ringOffsets},
    candidateCapacity: 65536, trackEventCapacity: 16,
    output: {
      events: {ids, status: {count, overflow, candidateOverflow: eventCandidateOverflow}, eventZones, eventTypes, eventTimes},
      dwellTimes, visitCounts,
      status: {candidateOverflow: candidateScratchOverflow}
    }
  }));
}
```

`output.events.eventPositions` (`float32x2`) holds the interpolated crossing position of each event.
`output.visitTable` is a sparse `(track, zone)` table that replaces the dense matrices for large
`trackCount * zoneCount`: rows are ordered by track, then zone, as a `GPUCompactOutput` (`ids` is the track,
plus `status.count`, `status.overflow` and `status.requiredCount`) extended with the columns `zones`, `visits`, `dwellTimes`,
`firstEnterTimes` and `lastExitTimes` (see `GPUZoneVisitTableOutput`). `addFleetDwellZoneEventsRecipe` forwards both outputs. Times are
relative to the track start, a track that starts inside has first enter 0, and an open visit has last exit
equal to the track duration. The table is built from the dense per-cell state, so it bounds the output, not
the working memory.

Optional `visitTable.status.candidateOverflow` reports candidate-scratch loss affecting the dense
visits, dwell and table. It does not include `trackEventCapacity` truncation: that cap only bounds
the displayed event list and leaves visits/dwell intact. Encounter and closest-approach compact
pair ports likewise propagate their hit/grid/entry loss through `pairs.status.candidateOverflow`.
The fleet-dwell recipe preserves caller event status views; its `zone-events` stage uses the
compact event candidate flag (scratch or track loss), while its separate scratch diagnostic
continues to identify scratch loss alone.

Limits: the scratch of `candidateCapacity` rows is sorted (three stable `GPUSort` passes) every encoding;
the walk and dwell kernels use one invocation per track; the dense matrices cost `trackCount * zoneCount`
rows; samples exactly on an edge follow the half-open rule and can disagree with other predicates by one
event; zones with overlapping rings are not supported (even-odd); at equal timestamps the crossing position is not unique.

Status: the optional `output.status` takes four one-row `uint32` views, `candidateCount`,
`candidateOverflow`, `trackOverflow` and `eventOverflow`.
- `candidateCount` is the unclamped number of segment-edge bounding-box candidates, including one ray query
  per track. Set `candidateCapacity` to at least this for exact results.
- `candidateOverflow` means the scratch was too small, so events may be missing.
- `trackOverflow` means a track has more than `trackEventCapacity` events. Its list is truncated, but dwell and
  visits stay exact.
- `eventOverflow` means there are more events than `output.events.ids.length`. `output.events.status.requiredCount`
  holds the size needed.

`output.events.status.overflow` reports `eventOverflow` only; optional
`output.events.status.candidateOverflow` reports the OR of candidate and track overflow.

An event-list readout is complete only when both event status flags are zero. Dwell totals,
visits and unclamped per-track counts instead require the scratch-only
`output.status.candidateOverflow` to be zero: an event-list cap alone does not invalidate them.
The standalone and fleet-dwell lenses capture these as `eventOverflow`,
`eventCandidateOverflow` and `candidateScratchOverflow`, with the two loss domains kept separate.
The website qualifies retained crossings as incomplete on either event-list flag, and qualifies
dwell only on scratch loss; pending status readbacks never certify completeness. The standalone
lens example accepts `candidateCapacity` (default 131072 scratch rows) independently of
`trackEventCapacity` (the maximum retained events per track).

#### How it computes

{/* lens:GPUZoneEvents */}

::lens{scenario=gpu-spatial-analysis.zone-events rung=datum datum=0 caption="Track 0 (MMSI 366993970): its zone events, dwell and visit counts"}

::lens{scenario=gpu-spatial-analysis.zone-events rung=kernel caption="The compute shaders that find crossings and sum dwell per track and zone"}

::lens{scenario=gpu-spatial-analysis.zone-events rung=graph caption="Command graph nodes and buffers that produce the events and dwell"}

{/* lens:addFleetDwellZoneEventsRecipe */}

::lens{scenario=gpu-spatial-analysis.fleet-dwell-zone-events-recipe rung=datum datum=0 caption="Anchorage 0 (Upper Bay Anchorage Number 20B): visiting tracks and total dwell"}

::lens{scenario=gpu-spatial-analysis.fleet-dwell-zone-events-recipe rung=kernel caption="The compute shaders that key dwell by zone for the roll-up"}

::lens{scenario=gpu-spatial-analysis.fleet-dwell-zone-events-recipe rung=graph caption="Command graph of the recipe: zone events, zone keys and zone statistics"}

### `GPUTrajectoryEncounters`

Candidate capacities reserve a strictly higher saturated-count sentinel; device bindings usually
impose much lower limits.

Finds pairs of tracks within a distance of each other in the same time bucket of resampled tracks.

::lens{scenario=gpu-spatial-analysis.trajectory-encounters rung=result caption="GPU pair table: 8 track pairs within 200 m at the same clock, smallest distance in metres"}

Pairs of tracks within `distance` in the same time bucket (MobilityDB `tdwithin` on resampled tracks).

Construction warns when the host-known `maximumDistance` derived from `cellSize` is too short for float32 lon/lat; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions) for this check and the unchecked per-frame `distance` GPU view.

Maximum: `columns * rows * bucketCount` of the lattice is at most `getGPUGridIndexMaximumCellCount(device)` (2^25 - 1 on a default device); raise `cellSize` or lower `bucketCount`.

- Inputs: dense `samples` (`[trackCount * bucketCount]` `float32x2`, row `track * bucketCount + bucket`,
  the `GPUTrajectoryResample.samples` layout), `trackCount`, `bucketCount`, optional `mask`,
  per-frame `distance` (one float row, clamped to `cellSize`), compile-time `cellSize` and
  `bounds [minX, minY, maxX, maxY]`, `hitCapacity` and optional `bucketTimes`.
- Outputs, all in `output`: `output.pairs` (a `GPUCompactOutput`, `ids` is the lower track, extended with the
  optional columns `partners`, `firstBuckets`, `minimumDistances`, `bucketCounts` and `firstTimes` (needs
  `bucketTimes`)) and the required `output.status.candidateOverflow`. Pairs have `track < partner`,
  are deduplicated across buckets and sorted by `(track, partner)`; the sentinel is `0xffffffff` (IDs,
  buckets) and 0 after `status.count`. `output.pairs.status.overflow` reports only pair-capacity truncation;
  `output.status.candidateOverflow` reports hit-scratch or grid incompleteness.
- It uses a 3D `GPUGridIndex` (x, y and the bucket as an exact third axis). Each sample maps its
  search distance into cell units around the builder's coordinate. Rounding padding is
  `max(1, ceil(axisCellCount * 64 * 2^-23))` cells on each side, independent of the absolute
  world coordinate. A planar 2048-by-2048 grid visits at most 9 cells at zero distance or 25 for a
  positive distance at most half a cell width, including translated projected coordinates.
  Only higher track IDs are visited, once per cell, including wrapped
  longitude images. This retains exact-distance pairs even when rounding separates them by
  two columns (for example -18.5 and -17.5 in [-20.5,20.5] with unit cells). It then uses
  three stable sorts of the hits and `GPUCompaction` for the runs.
- Great-circle: with `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`
  (optional `sphereRadius`) the samples are `(longitude, latitude)` degrees, `distance` and the reported
  `minimumDistances` are in the context's length unit (meters by default; kilometers with
  `units: 'kilometers'`), and `cellSize` and `bounds` stay in degrees. The default sphere radius is the
  mean Earth radius in that unit. It measures distance, so `local-lambert-equal-area` frames throw. The test is the
  great-circle distance of the two samples of one bucket, so there is no tangent-plane approximation. Pruning
  is conservative: two points within angle `theta` differ by at most `theta` in latitude and by at most
  `asin(sin(theta) / cos(L))` in longitude, where `L` is the largest `|latitude|` of `bounds`. The lattice
  therefore searches up to `maximumDistance = radius * asin(sin(cellSize) * cos(L))` (times `1 - 1e-6`),
  exposed as `encounters.maximumDistance`, and `distance` is clamped to it. Verified against pyproj: walking
  every azimuth at that distance from latitude ±L stays within one cell. Pick latitude bounds that exclude the
  polar caps (the reach shrinks to 0 as `L` approaches 90). Bounds with longitudes exactly `[-180, 180]` wrap
  the lattice across the antimeridian (sample longitudes are folded into that range first and `cellSize` must
  be below 120 degrees); other longitude bounds do not wrap, and `cellSize` must be below 90. Samples must
  have raw longitude separation at most 180 degrees to form a pair on a non-wrapping grid;
  this exclusion is independent of track order and candidate-range padding. Latitudes must
  lie in `(-90, 90)`. Planar contexts keep `maximumDistance = cellSize`. `addClockEncounters` forwards
  `spatialContext` to the resample and the encounters.
- It is discrete in time: the samples must share a clock (column `k` is one instant for all tracks), and
  encounters between buckets are missed. With `GPUTrajectoryResample` that holds only when the tracks cover
  the same window.

```ts
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUTrajectoryEncounters} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample10(graph: GPUCommandGraph, inputs: {
  samples: NonNullable<ConstructorParameters<typeof GPUTrajectoryEncounters>[0]['samples']>;
  trackCount: NonNullable<ConstructorParameters<typeof GPUTrajectoryEncounters>[0]['trackCount']>;
  bucketCount: NonNullable<ConstructorParameters<typeof GPUTrajectoryEncounters>[0]['bucketCount']>;
  distance: NonNullable<ConstructorParameters<typeof GPUTrajectoryEncounters>[0]['distance']>;
  ids: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryEncounters>[0]['output']>['pairs']>['ids']>;
  count: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryEncounters>[0]['output']>['pairs']>['status']>['count']>;
  overflow: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryEncounters>[0]['output']>['pairs']>['status']>['overflow']>;
  partners: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryEncounters>[0]['output']>['pairs']>['partners']>;
  firstBuckets: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryEncounters>[0]['output']>['pairs']>['firstBuckets']>;
  minimumDistances: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryEncounters>[0]['output']>['pairs']>['minimumDistances']>;
  candidateOverflow: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryEncounters>[0]['output']>['status']>['candidateOverflow']>;
}) {
  const {samples, trackCount, bucketCount, distance, ids, count, overflow, partners, firstBuckets, minimumDistances, candidateOverflow} = inputs;
  graph.add(new GPUTrajectoryEncounters({
    samples, trackCount, bucketCount, distance, cellSize: 50, bounds: [0, 0, 1000, 1000],
    hitCapacity: 1 << 16,
    output: {
      pairs: {ids, status: {count, overflow}, partners, firstBuckets, minimumDistances},
      status: {candidateOverflow}
    }
  }));
}
```

`addClockEncounters(graph, props)` builds the shared-clock chain in one call: it resamples all tracks
onto one clock (`GPUTrajectoryResample` with `spacing: 'clock'`, a per-frame `[start, step]` buffer) and
feeds `GPUTrajectoryEncounters`, so tracks need not cover the same window. It takes the trajectory layout
(`positions`, `timestamps`, `trackOffsets`) and the encounter options as flat props. Optional
`output.bucketTimes` and `output.samples` expose the clock and the resampled tracks
(`AddClockEncountersProps`, `ClockEncounters`). The optional per-frame `maximumGap` view is forwarded
to the clock resample, so a bucket inside an over-long track interval reads that track as absent and
produces no encounter.

Clock mode compares positions only at the shared bucket instants, so two tracks that pass each other
between buckets are missed. Treat it as a fast preview and use `GPUTrajectoryClosestApproach` for the
exact answer.

#### How it computes

{/* lens:GPUTrajectoryEncounters */}

::lens{scenario=gpu-spatial-analysis.trajectory-encounters rung=datum datum=0 caption="Pair 0: lower track, partner, buckets within 200 m and smallest distance"}

::lens{scenario=gpu-spatial-analysis.trajectory-encounters rung=formula caption="Bucket count and minimum distance over the clock buckets within range"}

::lens{scenario=gpu-spatial-analysis.trajectory-encounters rung=kernel caption="The compute shaders that scan the grid and reduce each pair"}

::lens{scenario=gpu-spatial-analysis.trajectory-encounters rung=graph caption="Command graph nodes and buffers that produce the pairs"}

### `GPUTrajectoryClosestApproach`

Candidate capacities reserve a strictly higher saturated-count sentinel; device bindings usually
impose much lower limits.

Finds pairs of tracks within a distance of each other at the same instant, with closest-approach distance, time and duration.

::lens{scenario=gpu-spatial-analysis.trajectory-closest-approach rung=result caption="GPU pair table: 10 track pairs passing within 200 m, closest approach and time inside"}

Exact encounters between trajectories on their raw samples, the MobilityDB `tdwithin` and
`nearestApproachDistance` question: tracks move linearly between samples, and two tracks are an encounter
while their positions at the same instant are at most `distance` apart. Each sample row becomes a segment to
the next sample of its track with a space-time box (`x` and `y` padded by half the distance), boxes are
binned into a hashed grid of `cellSize` by `cellSize` by `timeCellSize` cells and sorted, and every pair of
segments of different tracks that shares a cell is tested once (in the cell holding the minimum corner of
the two boxes' intersection). The test solves the closest point of approach over the common time interval:
the separation is `D + V tau`, its minimum is at `tau = -(D . V) / |V|^2`, and the time within the distance
lies between the roots of `|D + V tau|^2 = distance^2`. With the optional per-frame `maximumGap` view
(one `float32` row, the timestamps' unit) a segment longer than the limit is not motion: it adds no hits
and no cell entries, so a long gap can no longer raise `entryOverflow`.

- Inputs: the trajectory layout (`positions`, `timestamps` as `float32` values or `uint32x2` Int64 words in the declared timestamp unit,
  `trackOffsets`), optional `mask`, a per-frame one-row `distance`, and the compile-time `cellSize`,
  `timeCellSize`, `entryCapacity` (cell entries) and `hitCapacity` (segment-pair hits). Planar only; project
  longitude/latitude first with `createGPULocalFrame`. `timeCellSize` uses the declared timestamp unit,
  so rescale it with timestamps when changing units.
- Outputs, in `output`: `pairs`, a `GPUCompactOutput` (`ids` is the lower track) in `(track, partner)`
  order with optional `partners`, `minimumDistances`, `closestTimes`, `firstTimes` and `durations`, all
  times in the declared timestamp unit relative to the first sample of the lower track, with durations in that same unit; and `status.entryOverflow` and
  `status.hitOverflow` for the two scratch capacities (a segment covering more than 4096 cells also sets
  `entryOverflow`).
- Word timestamps stay exact: times are measured from a segment's own start row before the f32
  conversion. Verified against `test/oracles/trajectory-closest-approach.py`, a float64 brute force over
  every segment pair, self-checked by dense sampling and against shapely path distances.

```ts
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUTrajectoryClosestApproach} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample11(graph: GPUCommandGraph, inputs: {
  positions: NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['positions']>;
  timestamps: NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['timestamps']>;
  trackOffsets: NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['trackOffsets']>;
  distance: NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['distance']>;
  ids: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['output']>['pairs']>['ids']>;
  count: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['output']>['pairs']>['status']>['count']>;
  overflow: NonNullable<NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['output']>['pairs']>['status']>['overflow']>;
  partners: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['output']>['pairs']>['partners']>;
  minimumDistances: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['output']>['pairs']>['minimumDistances']>;
  closestTimes: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['output']>['pairs']>['closestTimes']>;
  firstTimes: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['output']>['pairs']>['firstTimes']>;
  durations: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['output']>['pairs']>['durations']>;
  entryOverflow: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['output']>['status']>['entryOverflow']>;
  hitOverflow: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryClosestApproach>[0]['output']>['status']>['hitOverflow']>;
}) {
  const {positions, timestamps, trackOffsets, distance, ids, count, overflow, partners, minimumDistances, closestTimes, firstTimes, durations, entryOverflow, hitOverflow} = inputs;
  graph.add(new GPUTrajectoryClosestApproach({
    positions, timestamps, trackOffsets, distance,
    cellSize: 50, timeCellSize: 60, entryCapacity: 1 << 20, hitCapacity: 1 << 18,
    output: {
      pairs: {ids, status: {count, overflow}, partners, minimumDistances, closestTimes, firstTimes, durations},
      status: {entryOverflow, hitOverflow}
    }
  }));
}
```

#### How it computes

{/* lens:GPUTrajectoryClosestApproach */}

::lens{scenario=gpu-spatial-analysis.trajectory-closest-approach rung=datum datum=0 caption="Pair 0: tracks, closest-approach distance, its time and the time within 200 m"}

::lens{scenario=gpu-spatial-analysis.trajectory-closest-approach rung=formula caption="Closest time on a common segment interval and the distance at that time"}

::lens{scenario=gpu-spatial-analysis.trajectory-closest-approach rung=kernel caption="The compute shaders that test segment pairs and reduce each track pair"}

::lens{scenario=gpu-spatial-analysis.trajectory-closest-approach rung=graph caption="Command graph nodes and buffers that produce the pairs"}

### `GPUTrackSimilarity`

Multipart operands are supported for Hausdorff, maximum distance and SSPD; pairs index logical
features and distances never bridge disconnected parts. Sequence metrics require a caller-chosen
single-part traversal; see [multipart semantics and limits](./track-similarity-measures.md).

invalidPair also reports invalid track ranges and densification overflow; an invalid pair must not
be interpreted as a valid similarity.

Measures Hausdorff and discrete Frechet distance between pairs of tracks or rings, with optional maximum distance and dynamic time warping cost.

::lens{scenario=gpu-spatial-analysis.track-similarity rung=result caption="All 66 track pairs: Hausdorff and Fréchet distance in metres"}

Hausdorff (symmetric, vertex-to-segment as GEOS computes it, unbounded length) and discrete Frechet
(vertex-based) distances for given pairs of tracks or rings (`hausdorff_distance(a, b, densify)`,
`frechet_distance(a, b, densify)`). Behavior change: Hausdorff used to be vertex-to-vertex, which agreed
with Shapely only by coincidence; it now measures from the (densified) vertices to the other polyline's
segments, so values can be slightly smaller than before. Inputs are `linesA` (a tagged line geometry:
`kind: 'lines'`, `positions` and `lineOffsets`; optionally `linesB`, by
default the same set), `pairA`, `pairB` (`uint32` views) and an optional GPU-written `activePairCount`.
Outputs, all in `output`, are `hausdorff`, `frechet`, optional `maximumDistance` (largest vertex-to-vertex distance per pair, Sedona
`ST_MaxDistance`, original vertices) and `pairStatus` (bits of `GPU_TRACK_SIMILARITY_STATUS`: `emptyTrack` 1,
`frechetCapExceeded` 2, `invalidPair` 4, `invalidGeometry` 8, `invalidDecisionThreshold` 16).
Invalid/absent/inactive pairs report bit 4 alone. On addressable pairs, invalid thresholds can
compose with empty-track or invalid-geometry evidence; invalid geometry suppresses cap evidence.
Empty/invalid geometry makes all distances NaN. Cap evidence makes Fréchet, DTW, EDR, ERP and
LCSS unavailable while Hausdorff, SSPD and maximum distance remain usable. Bit 16 affects only
`frechetWithin`: distances remain usable. Unavailable decisions are zero, so use
`decodeGPUTrackSimilarityPairStatus` and its per-measure validity fields before interpreting them.
`densify` (0 or in `[1/4096, 1]`, compile-time) splits each segment into `round(1/densify)` parts, as
Shapely does; the virtual vertices are computed on the fly. Frechet sweeps the longer track in strips of
256 lanes, so only the shorter (densified) track is capped: `maximumFrechetVertexCount` is 1 to 2048 (default
256). A pair over the cap reports `frechetCapExceeded` with a NaN Frechet while its Hausdorff is still
computed. Polygon boundaries are passed as closed vertex lists. Against Shapely 2.1.2 (GEOS 3.13.1) all
three measures were within 2e-4 relative on line and polygon-boundary pairs, including a 700 by 300 Frechet
over three strips. Rings are open vertex
sequences, so Frechet is sensitive to the start and the orientation. Pairs can come straight from
`GPUTrajectoryEncounters` (`output.pairs.ids`, `output.pairs.partners` and `output.pairs.status.count`).

For `frechetEpsilon`, no warning is possible because the distance is a GPU view; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

Maximum: the pair capacity `pairA.length` is at most `GPU_TRACK_SIMILARITY_MAXIMUM_PAIR_COUNT` (2^24 - 1).

`spatialContext` defaults to native planar (Euclidean distances in the f32 input coordinates; the planar kernels
compile no geodesic code). `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` makes Hausdorff,
Frechet and `maximumDistance` great-circle distances in the context's length unit (meters by default;
kilometers with `units: 'kilometers'`) on a sphere of `sphereRadius` (default the mean Earth radius in
that unit). Other geographic metrics throw. It measures distance, so `local-lambert-equal-area` frames throw. Vertices are `[longitude, latitude]` degrees
and segments are straight in longitude/latitude, as everywhere in this package. The vertex-to-segment distance is
the haversine distance to the closest point of the segment, with that point found in the local equirectangular
frame of the vertex (`geodesicSegmentFraction`, `geodesicSegmentDistance`): exact to first order for segments of
kilometers, so `densify` long segments. Densified points interpolate longitude and latitude linearly with the
longitude difference wrapped to +-180, so tracks may cross the antimeridian. Against numpy haversine with a
20000-part segment densification (cross-checked with `pyproj.Geod(a=6371008.8, f=0)`) the f32 results are
expected within 2e-3 relative; the spec covers a mid-latitude, a 66 degree and an antimeridian pair.

```ts
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUTrackSimilarity} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample12(graph: GPUCommandGraph, inputs: {
  positions: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['linesA']>['positions']>;
  lineOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['linesA']>['lineOffsets']>;
  pairs: {ids: GraphDataView<'uint32'>; partners: GraphDataView<'uint32'>; status: {count: GraphDataView<'uint32'>}};
  hausdorff: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['output']>['hausdorff']>;
  frechet: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['output']>['frechet']>;
  maximumDistance: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['output']>['maximumDistance']>;
  pairStatus: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['output']>['pairStatus']>;
}) {
  const {positions, lineOffsets, pairs, hausdorff, frechet, maximumDistance, pairStatus} = inputs;
  graph.add(new GPUTrackSimilarity({
    linesA: {kind: 'lines', positions, lineOffsets},
    pairA: pairs.ids, pairB: pairs.partners, activePairCount: pairs.status.count,
    maximumFrechetVertexCount: 256, densify: 0.25,
    output: {hausdorff, frechet, maximumDistance, pairStatus}
  }));
}
```

Two more measures share the Frechet layout and its shorter-track cap. `output.frechetWithin` (`uint32`,
with the per-frame one-row `frechetEpsilon` in the context's length unit) is the Frechet decision, 1 when the discrete Frechet distance
is at most epsilon: endpoint distances reject pairs before scanning their vertices; a fixed monotone coupling
accepts pairs in `O(n + m)`. The remaining pairs run the boolean free-space program,
evaluating membership only for cells with reachable predecessors; it stops once no
monotone path can survive a row or two consecutive anti-diagonals. Use it to filter candidate pairs at a
threshold without requesting the full distance or DTW, which still compute when requested alongside the decision.
These are the base contributor's early decisions. `GPUTrackSimilaritySearch` separately applies
metric-specific search bounds before delegating retained pairs: endpoint lower bounds for Fréchet
and DTW, possible-match upper bounds for LCSS, and planar box bounds for Hausdorff, SSPD and EDR.
Geographic Hausdorff and SSPD use a zero lower bound; geographic EDR uses `abs(n - m)`, the
absolute vertex-count difference. ERP uses the difference of gap-distance sums in both contexts.
For separated geographic singleton tracks, EDR's lower bound is 0 even when its edit distance is 1.
See [the search bounds and stored-coordinate membership contract](./track-similarity-measures.md#definitions).
Planar membership is inclusive on stored coordinates; a rounded displayed Fréchet distance of 1
can coexist with `frechetWithin=0` at epsilon 1, including `(-2^-25,0) → (1,0)`.
`track-similarity-dtw-frechet-ab.spec.ts` times decision-only and distance-only graphs separately.
`output.dynamicTimeWarping` is the dynamic time warping cost, the
smallest sum of point distances in the context's length unit over monotone warping paths (`similaritymeasures.dtw`; not the square root
of summed squares of tslearn and dtaidistance), with an optional compile-time Sakoe-Chiba band
`dynamicTimeWarpingBandRadius` equal to `tslearn.metrics.sakoe_chiba_mask(n, m, radius)`; cells outside
the band are never evaluated. Both are verified against `test/oracles/track-similarity-dtw-frechet.py`
(similaritymeasures, shapely, tslearn), planar and great-circle.

```ts
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUTrackSimilarity} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addOperatorsLinesExample13(graph: GPUCommandGraph, inputs: {
  positions: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['linesA']>['positions']>;
  lineOffsets: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['linesA']>['lineOffsets']>;
  pairs: {ids: GraphDataView<'uint32'>; partners: GraphDataView<'uint32'>; status: {count: GraphDataView<'uint32'>}};
  frechetEpsilon: NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['frechetEpsilon']>;
  frechetWithin: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['output']>['frechetWithin']>;
  pairStatus: GraphDataView<'uint32'>;
  dynamicTimeWarping: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['output']>['dynamicTimeWarping']>;
}) {
  const {positions, lineOffsets, pairs, frechetEpsilon, frechetWithin, dynamicTimeWarping, pairStatus} = inputs;
  graph.add(new GPUTrackSimilarity({
    linesA: {kind: 'lines', positions, lineOffsets},
    pairA: pairs.ids, pairB: pairs.partners, activePairCount: pairs.status.count,
    frechetEpsilon, dynamicTimeWarpingBandRadius: 8,
    output: {frechetWithin, dynamicTimeWarping, pairStatus}
  }));
}
```

#### How it computes

{/* lens:GPUTrackSimilarity */}

::lens{scenario=gpu-spatial-analysis.track-similarity rung=datum datum=0 caption="Pair 0 (tracks 0 and 1): Hausdorff, Frechet and maximum vertex distance"}

::lens{scenario=gpu-spatial-analysis.track-similarity rung=kernel caption="The compute shaders that measure each pair"}

::lens{scenario=gpu-spatial-analysis.track-similarity rung=graph caption="Command graph nodes and buffers that produce the pair distances"}


At the resample-to-cell handoff, select `longitudePolicy: 'wrap'` on the downstream
`GPUPointToCell` to handle longitudes unwrapped beyond ±180°. The family defaults differ; see
[the five-family longitude table](./operators-cells.md#cell-system-support). Use the canonical
`precision: 'double-single'` spelling where supported.

Base similarity status includes bit 8 `invalidGeometry` and bit 16 `invalidDecisionThreshold`.
Always request `pairStatus` beside decisions and call `decodeGPUTrackSimilarityPairStatus` on
readback. `decisionValid` must be true before treating `frechetWithin: 0` as rejection; identical
tracks with epsilon −1 return distance 0, decision 0, status 16. `activePairCount` identifies the
base contributor's inactive suffix, which shares the invalid-pair bit with bad indices.


```ts
import {decodeGPUTrackSimilarityPairStatus} from '@luma-contributors/analysis/gpu-spatial-analysis';
export function readFrechetDecisions(pairStatus: Uint32Array | Uint8Array | ArrayBuffer) {
  return decodeGPUTrackSimilarityPairStatus(pairStatus); // inspect each row's decisionValid
}
```
