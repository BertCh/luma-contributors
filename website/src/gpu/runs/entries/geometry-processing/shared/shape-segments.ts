// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Adapter kernels that turn an operation's shape rows (circles, bounding boxes, rotated rectangles)
// into line segments (float32x4) the `lines` layer can draw. They are extra nodes in the same
// command graph, so the picture still reads the buffers the operation wrote on the GPU.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {GPUGeodesicDestination} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

const CIRCLE_SEGMENT_COUNT = 40;

type KernelBinding = {name: string; buffer: RunBuffer; access: 'read' | 'read_write'};

function getElementType(buffer: RunBuffer): 'f32' | 'u32' | 'i32' {
  if (buffer.format.startsWith('uint32')) return 'u32';
  if (buffer.format.startsWith('sint32')) return 'i32';
  return 'f32';
}

/** Adds one linear WGSL kernel node to the run's graph. */
export function addKernel(
  builder: RunBuilder,
  options: {id: string; invocationCount: number; bindings: KernelBinding[]; body: string}
): void {
  builder.graph.add({
    getCommandNodes: <Parameters>(
      graph: GPUCommandGraph<Parameters>
    ): GPUCommandNode<Parameters>[] => [
      createWGSLKernelNode<Parameters>(graph, {
        id: options.id,
        operation: 'GeometryRunAdapter',
        bindings: options.bindings.map(binding => ({
          name: binding.name,
          view: binding.buffer.view,
          type: getElementType(binding.buffer),
          access: binding.access
        })),
        invocationCount: options.invocationCount,
        body: options.body
      })
    ]
  });
}

/** Circle outlines (`circles` rows are center x, center y, radius, ...) as segments. */
export function addCircleSegments(
  builder: RunBuilder,
  id: string,
  circles: RunBuffer<'float32x3' | 'float32x4'>
): RunBuffer<'float32x4'> {
  const stride = circles.format === 'float32x3' ? 3 : 4;
  const segments = builder.allocate(
    `${id}-segments`,
    'float32x4',
    circles.length * CIRCLE_SEGMENT_COUNT
  );
  addKernel(builder, {
    id: `${id}-circle-segments`,
    invocationCount: circles.length * CIRCLE_SEGMENT_COUNT,
    bindings: [
      {name: 'circles', buffer: circles, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let circle = index / ${CIRCLE_SEGMENT_COUNT}u;
  let step = index % ${CIRCLE_SEGMENT_COUNT}u;
  let base = circlesOffset + circle * ${stride}u;
  let center = vec2<f32>(circles[base], circles[base + 1u]);
  let radius = circles[base + 2u];
  let angle0 = f32(step) * ${(Math.PI * 2) / CIRCLE_SEGMENT_COUNT};
  let angle1 = f32(step + 1u) * ${(Math.PI * 2) / CIRCLE_SEGMENT_COUNT};
  let outBase = segmentsOffset + index * 4u;
  segments[outBase] = center.x + cos(angle0) * radius;
  segments[outBase + 1u] = center.y + sin(angle0) * radius;
  segments[outBase + 2u] = center.x + cos(angle1) * radius;
  segments[outBase + 3u] = center.y + sin(angle1) * radius;`
  });
  return segments;
}

/** Axis-aligned boxes (`[minX, minY, maxX, maxY]` rows) as four segments each. */
export function addBoxSegments(
  builder: RunBuilder,
  id: string,
  boxes: RunBuffer<'float32x4'>
): RunBuffer<'float32x4'> {
  return addBoxSegmentsOfRows(builder, id, boxes, boxes.length);
}

/** Like addBoxSegments for any float buffer whose first `rowCount` rows of four floats are boxes. */
export function addBoxSegmentsOfRows(
  builder: RunBuilder,
  id: string,
  boxes: RunBuffer,
  rowCount: number
): RunBuffer<'float32x4'> {
  const segments = builder.allocate(`${id}-segments`, 'float32x4', rowCount * 4);
  addKernel(builder, {
    id: `${id}-box-segments`,
    invocationCount: rowCount * 4,
    bindings: [
      {name: 'boxes', buffer: boxes, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let boxIndex = index / 4u;
  let side = index % 4u;
  let base = boxesOffset + boxIndex * 4u;
  let low = vec2<f32>(boxes[base], boxes[base + 1u]);
  let high = vec2<f32>(boxes[base + 2u], boxes[base + 3u]);
  var fromPoint = low;
  var toPoint = vec2<f32>(high.x, low.y);
  if (side == 1u) { fromPoint = toPoint; toPoint = high; }
  if (side == 2u) { fromPoint = high; toPoint = vec2<f32>(low.x, high.y); }
  if (side == 3u) { fromPoint = vec2<f32>(low.x, high.y); toPoint = low; }
  let outBase = segmentsOffset + index * 4u;
  segments[outBase] = fromPoint.x;
  segments[outBase + 1u] = fromPoint.y;
  segments[outBase + 2u] = toPoint.x;
  segments[outBase + 3u] = toPoint.y;`
  });
  return segments;
}

/** Quadrilaterals given as four consecutive corner rows (float32x2) per group, as four segments. */
export function addQuadSegments(
  builder: RunBuilder,
  id: string,
  corners: RunBuffer<'float32x2'>
): RunBuffer<'float32x4'> {
  const segments = builder.allocate(`${id}-segments`, 'float32x4', corners.length);
  addKernel(builder, {
    id: `${id}-quad-segments`,
    invocationCount: corners.length,
    bindings: [
      {name: 'corners', buffer: corners, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let groupIndex = index / 4u;
  let side = index % 4u;
  let fromBase = cornersOffset + (groupIndex * 4u + side) * 2u;
  let toBase = cornersOffset + (groupIndex * 4u + (side + 1u) % 4u) * 2u;
  let outBase = segmentsOffset + index * 4u;
  segments[outBase] = corners[fromBase];
  segments[outBase + 1u] = corners[fromBase + 1u];
  segments[outBase + 2u] = corners[toBase];
  segments[outBase + 3u] = corners[toBase + 1u];`
  });
  return segments;
}

const FAR_AWAY = 1e6;

/** Copies the first `count` rows of `points` and parks the remaining rows far outside the view. */
export function addMaskedPoints(
  builder: RunBuilder,
  id: string,
  points: RunBuffer<'float32x2' | 'float32x4'>,
  count: RunBuffer<'uint32'>
): RunBuffer<'float32x2'> {
  // A float32x4 row (a segment) holds two points, and `count` counts rows.
  const pointsPerRow = points.format === 'float32x4' ? 2 : 1;
  const pointCount = points.length * pointsPerRow;
  const masked = builder.allocate(`${id}-masked`, 'float32x2', pointCount);
  addKernel(builder, {
    id: `${id}-mask-points`,
    invocationCount: pointCount,
    bindings: [
      {name: 'points', buffer: points, access: 'read'},
      {name: 'count', buffer: count, access: 'read'},
      {name: 'masked', buffer: masked, access: 'read_write'}
    ],
    body: `let isLive = index / ${pointsPerRow}u < count[countOffset];
  masked[maskedOffset + index * 2u] = select(${FAR_AWAY}, points[pointsOffset + index * 2u], isLive);
  masked[maskedOffset + index * 2u + 1u] = select(${FAR_AWAY}, points[pointsOffset + index * 2u + 1u], isLive);`
  });
  return masked;
}

/**
 * Spreads a per-row category onto the two endpoint nodes of every segment row (the endpoints
 * buffer read as nodes: row r owns nodes 2r and 2r + 1). Rows at or past `count` get the
 * unreached value, so the `segments` layer leaves them out.
 */
export function addEndpointCategories(
  builder: RunBuilder,
  id: string,
  categories: RunBuffer<'uint32'>,
  count: RunBuffer<'uint32'>,
  rowCount: number
): RunBuffer<'uint32'> {
  const nodeCategories = builder.allocate(`${id}-node-categories`, 'uint32', rowCount * 2);
  addKernel(builder, {
    id: `${id}-endpoint-categories`,
    invocationCount: rowCount * 2,
    bindings: [
      {name: 'categories', buffer: categories, access: 'read'},
      {name: 'count', buffer: count, access: 'read'},
      {name: 'nodeCategories', buffer: nodeCategories, access: 'read_write'}
    ],
    body: `let row = index / 2u;
  nodeCategories[nodeCategoriesOffset + index] = select(0xffffffffu, categories[categoriesOffset + row], row < count[countOffset]);`
  });
  return nodeCategories;
}

/**
 * Edge pairs (uint32x2, one per output vertex) for an output whose lines or rings are given by
 * GPU-written `offsets`: vertex i links to i + 1 inside its line, and (when `isClosed`) the last
 * vertex of a ring links back to the first. Vertices outside every line get a degenerate edge.
 */
export function addOutputEdges(
  builder: RunBuilder,
  id: string,
  offsets: RunBuffer<'uint32'>,
  vertexCapacity: number,
  isClosed: boolean
): RunBuffer<'uint32x2'> {
  const edges = builder.allocate(`${id}-edges`, 'uint32x2', vertexCapacity);
  addKernel(builder, {
    id: `${id}-output-edges`,
    invocationCount: vertexCapacity,
    bindings: [
      {name: 'offsets', buffer: offsets, access: 'read'},
      {name: 'edges', buffer: edges, access: 'read_write'}
    ],
    body: `var nextVertex = index;
  for (var line = 0u; line + 1u < ${offsets.length}u; line++) {
    let start = offsets[offsetsOffset + line];
    let end = offsets[offsetsOffset + line + 1u];
    if (index >= start && index < end) {
      if (index + 1u < end) { nextVertex = index + 1u; }
      ${isClosed ? 'else { nextVertex = start; }' : ''}
    }
  }
  edges[edgesOffset + index * 2u] = index;
  edges[edgesOffset + index * 2u + 1u] = nextVertex;`
  });
  return edges;
}

/** Copies segment rows and parks all-zero (unwritten) rows far outside the view. */
export function addWrittenSegments(
  builder: RunBuilder,
  id: string,
  segments: RunBuffer<'float32x4'>
): RunBuffer<'float32x4'> {
  const written = builder.allocate(`${id}-written`, 'float32x4', segments.length);
  addKernel(builder, {
    id: `${id}-written-segments`,
    invocationCount: segments.length,
    bindings: [
      {name: 'segments', buffer: segments, access: 'read'},
      {name: 'written', buffer: written, access: 'read_write'}
    ],
    body: `let base = segmentsOffset + index * 4u;
  let isWritten = segments[base] != 0.0 || segments[base + 1u] != 0.0 || segments[base + 2u] != 0.0 || segments[base + 3u] != 0.0;
  for (var component = 0u; component < 4u; component++) {
    written[writtenOffset + index * 4u + component] = select(${FAR_AWAY}, segments[base + component], isWritten);
  }`
  });
  return written;
}

/** Spreads a per-feature uint32 onto its vertices (`featureOfVertex` is a static uint32 buffer). */
export function addFeatureValues(
  builder: RunBuilder,
  id: string,
  featureValues: RunBuffer<'uint32'>,
  featureOfVertex: RunBuffer<'uint32'>
): RunBuffer<'uint32'> {
  const vertexValues = builder.allocate(`${id}-vertex-values`, 'uint32', featureOfVertex.length);
  addKernel(builder, {
    id: `${id}-feature-values`,
    invocationCount: featureOfVertex.length,
    bindings: [
      {name: 'featureValues', buffer: featureValues, access: 'read'},
      {name: 'featureOfVertex', buffer: featureOfVertex, access: 'read'},
      {name: 'vertexValues', buffer: vertexValues, access: 'read_write'}
    ],
    body: 'vertexValues[vertexValuesOffset + index] = featureValues[featureValuesOffset + featureOfVertex[featureOfVertexOffset + index]];'
  });
  return vertexValues;
}

/** Maps longitude/latitude degrees to the drawing world (equirectangular, small margin). */
export function addEquirectangular(
  builder: RunBuilder,
  id: string,
  lonLat: RunBuffer<'float32x2'>,
  world: {width: number; height: number}
): RunBuffer<'float32x2'> {
  const projected = builder.allocate(`${id}-world`, 'float32x2', lonLat.length);
  addKernel(builder, {
    id: `${id}-equirectangular`,
    invocationCount: lonLat.length,
    bindings: [
      {name: 'lonLat', buffer: lonLat, access: 'read'},
      {name: 'projected', buffer: projected, access: 'read_write'}
    ],
    body: `let lon = lonLat[lonLatOffset + index * 2u];
  let lat = lonLat[lonLatOffset + index * 2u + 1u];
  projected[projectedOffset + index * 2u] = ${world.width * 0.04} + (lon + 180.0) / 360.0 * ${world.width * 0.92};
  projected[projectedOffset + index * 2u + 1u] = 0.04 + (lat + 90.0) / 180.0 * 0.92;`
  });
  return projected;
}

/** Segments (x0, y0, x1, y1) between aligned rows of two point buffers. */
export function addPairSegments(
  builder: RunBuilder,
  id: string,
  starts: RunBuffer<'float32x2'>,
  ends: RunBuffer<'float32x2'>
): RunBuffer<'float32x4'> {
  const segments = builder.allocate(`${id}-pairs`, 'float32x4', starts.length);
  addKernel(builder, {
    id: `${id}-pair-segments`,
    invocationCount: starts.length,
    bindings: [
      {name: 'starts', buffer: starts, access: 'read'},
      {name: 'ends', buffer: ends, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let base = segmentsOffset + index * 4u;
  segments[base] = starts[startsOffset + index * 2u];
  segments[base + 1u] = starts[startsOffset + index * 2u + 1u];
  segments[base + 2u] = ends[endsOffset + index * 2u];
  segments[base + 3u] = ends[endsOffset + index * 2u + 1u];`
  });
  return segments;
}

/** The ring number of every output vertex row (unreached rows get 0xffffffff), for coloring rings. */
export function addRingCategories(
  builder: RunBuilder,
  id: string,
  offsets: RunBuffer<'uint32'>,
  vertexCapacity: number
): RunBuffer<'uint32'> {
  const categories = builder.allocate(`${id}-ring-categories`, 'uint32', vertexCapacity);
  addKernel(builder, {
    id: `${id}-ring-categories`,
    invocationCount: vertexCapacity,
    bindings: [
      {name: 'offsets', buffer: offsets, access: 'read'},
      {name: 'categories', buffer: categories, access: 'read_write'}
    ],
    body: `var category = 0xffffffffu;
  for (var ring = 0u; ring + 1u < ${offsets.length}u; ring++) {
    if (index >= offsets[offsetsOffset + ring] && index < offsets[offsetsOffset + ring + 1u]) { category = ring; }
  }
  categories[categoriesOffset + index] = category;`
  });
  return categories;
}

/** Scales and shifts positions in the drawing world, to lay several panels side by side. */
export function addPanel(
  builder: RunBuilder,
  id: string,
  positions: RunBuffer<'float32x2'>,
  panel: {scaleX: number; scaleY: number; offsetX: number; offsetY: number}
): RunBuffer<'float32x2'> {
  const placed = builder.allocate(`${id}-panel`, 'float32x2', positions.length);
  addKernel(builder, {
    id: `${id}-panel-kernel`,
    invocationCount: positions.length,
    bindings: [
      {name: 'positions', buffer: positions, access: 'read'},
      {name: 'placed', buffer: placed, access: 'read_write'}
    ],
    body: `placed[placedOffset + index * 2u] = positions[positionsOffset + index * 2u] * ${panel.scaleX} + ${panel.offsetX};
  placed[placedOffset + index * 2u + 1u] = positions[positionsOffset + index * 2u + 1u] * ${panel.scaleY} + ${panel.offsetY};`
  });
  return placed;
}

const LATTICE_CAPACITY = 640;

/**
 * The precision lattice as full-length grid lines (x0, y0, x1, y1) at multiples of `gridSize`,
 * padded to a fixed row count with parked rows so a per-frame `gridSize` can be written into the
 * same buffer. Lattices denser than the capacity keep every n-th line.
 */
export function createLatticeSegments(
  gridSize: number,
  world: {width: number; height: number}
): Float32Array {
  const segments = new Float32Array(LATTICE_CAPACITY * 4).fill(FAR_AWAY);
  const columns = Math.floor(world.width / gridSize);
  const rows = Math.floor(world.height / gridSize);
  const stride = Math.max(1, Math.ceil((columns + rows + 2) / LATTICE_CAPACITY));
  let row = 0;
  for (let column = 0; column <= columns && row < LATTICE_CAPACITY; column += stride) {
    segments.set([column * gridSize, 0, column * gridSize, world.height], row++ * 4);
  }
  for (let line = 0; line <= rows && row < LATTICE_CAPACITY; line += stride) {
    segments.set([0, line * gridSize, world.width, line * gridSize], row++ * 4);
  }
  return segments;
}

/** A buffer of lattice lines for the `lines` layer; rewrite it with `write(createLatticeSegments(...))`. */
export function allocateLattice(
  builder: RunBuilder,
  id: string,
  gridSize: number,
  world: {width: number; height: number}
): RunBuffer<'float32x4'> {
  return builder.upload(id, 'float32x4', createLatticeSegments(gridSize, world));
}

/**
 * Great-circle arcs from the real GPUGeodesicDestination: every row (origin, bearing, distance) is
 * sampled at k/steps of its distance, and the samples are joined into world-space segments for the
 * `lines` layer. Segments that wrap across the date line are parked outside the view.
 */
export function addGeodesicArcs(
  builder: RunBuilder,
  id: string,
  options: {
    origins: RunBuffer<'float32x2'>;
    bearings: RunBuffer<'float32'>;
    distances: RunBuffer<'float32'>;
    rowCount: number;
    steps: number;
    world: {width: number; height: number};
  }
): RunBuffer<'float32x4'> {
  const {origins, bearings, distances, rowCount, steps, world} = options;
  const sampleCount = rowCount * steps;
  const sampleOrigins = builder.allocate(`${id}-sample-origins`, 'float32x2', sampleCount);
  const sampleBearings = builder.allocate(`${id}-sample-bearings`, 'float32', sampleCount);
  const sampleDistances = builder.allocate(`${id}-sample-distances`, 'float32', sampleCount);
  addKernel(builder, {
    id: `${id}-arc-samples`,
    invocationCount: sampleCount,
    bindings: [
      {name: 'origins', buffer: origins, access: 'read'},
      {name: 'bearings', buffer: bearings, access: 'read'},
      {name: 'distances', buffer: distances, access: 'read'},
      {name: 'sampleOrigins', buffer: sampleOrigins, access: 'read_write'},
      {name: 'sampleBearings', buffer: sampleBearings, access: 'read_write'},
      {name: 'sampleDistances', buffer: sampleDistances, access: 'read_write'}
    ],
    body: `let row = index / ${steps}u;
  let step = index % ${steps}u;
  sampleOrigins[sampleOriginsOffset + index * 2u] = origins[originsOffset + row * 2u];
  sampleOrigins[sampleOriginsOffset + index * 2u + 1u] = origins[originsOffset + row * 2u + 1u];
  sampleBearings[sampleBearingsOffset + index] = bearings[bearingsOffset + row];
  sampleDistances[sampleDistancesOffset + index] = distances[distancesOffset + row] * f32(step + 1u) / ${steps}.0;`
  });
  const samples = builder.allocate(`${id}-samples`, 'float32x2', sampleCount);
  builder.graph.add(
    new GPUGeodesicDestination({
      id: `${id}-geodesic`,
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'ellipsoidal',
        units: 'meters'
      },
      origins: sampleOrigins.view,
      bearings: sampleBearings.view,
      distances: sampleDistances.view,
      output: {
        destinations: samples.view,
        finalBearings: builder.allocate(`${id}-sample-final-bearings`, 'float32', sampleCount).view
      }
    })
  );
  const segments = builder.allocate(`${id}-arc-segments`, 'float32x4', sampleCount);
  addKernel(builder, {
    id: `${id}-arc-segments`,
    invocationCount: sampleCount,
    bindings: [
      {name: 'origins', buffer: origins, access: 'read'},
      {name: 'samples', buffer: samples, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let row = index / ${steps}u;
  let step = index % ${steps}u;
  var startPoint = vec2<f32>(origins[originsOffset + row * 2u], origins[originsOffset + row * 2u + 1u]);
  if (step > 0u) {
    startPoint = vec2<f32>(samples[samplesOffset + (index - 1u) * 2u], samples[samplesOffset + (index - 1u) * 2u + 1u]);
  }
  let endPoint = vec2<f32>(samples[samplesOffset + index * 2u], samples[samplesOffset + index * 2u + 1u]);
  let isWrapped = abs(startPoint.x - endPoint.x) > 180.0;
  let base = segmentsOffset + index * 4u;
  segments[base] = select(${world.width * 0.04} + (startPoint.x + 180.0) / 360.0 * ${world.width * 0.92}, ${FAR_AWAY}, isWrapped);
  segments[base + 1u] = select(0.04 + (startPoint.y + 90.0) / 180.0 * 0.92, ${FAR_AWAY}, isWrapped);
  segments[base + 2u] = select(${world.width * 0.04} + (endPoint.x + 180.0) / 360.0 * ${world.width * 0.92}, ${FAR_AWAY}, isWrapped);
  segments[base + 3u] = select(0.04 + (endPoint.y + 90.0) / 180.0 * 0.92, ${FAR_AWAY}, isWrapped);`
  });
  return segments;
}

/**
 * A small arrow on the first edge of every ring (three segments each: shaft and two wings), so
 * the winding direction can be read. `positions` holds the rings that `ringOffsets` delimits.
 */
export function addRingArrows(
  builder: RunBuilder,
  id: string,
  positions: RunBuffer<'float32x2'>,
  ringOffsets: RunBuffer<'uint32'>,
  ringCount: number
): RunBuffer<'float32x4'> {
  const segments = builder.allocate(`${id}-arrows`, 'float32x4', ringCount * 3);
  addKernel(builder, {
    id: `${id}-ring-arrows`,
    invocationCount: ringCount,
    bindings: [
      {name: 'positions', buffer: positions, access: 'read'},
      {name: 'offsets', buffer: ringOffsets, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let first = offsets[offsetsOffset + index];
  let start = vec2<f32>(positions[positionsOffset + first * 2u], positions[positionsOffset + first * 2u + 1u]);
  let end = vec2<f32>(positions[positionsOffset + first * 2u + 2u], positions[positionsOffset + first * 2u + 3u]);
  let direction = normalize(end - start + vec2<f32>(1e-9, 0.0));
  let normal = vec2<f32>(-direction.y, direction.x);
  let middle = (start + end) * 0.5;
  let tip = middle + direction * 0.024;
  let tail = middle - direction * 0.024;
  let wingA = tip - direction * 0.022 + normal * 0.013;
  let wingB = tip - direction * 0.022 - normal * 0.013;
  let base = segmentsOffset + index * 12u;
  segments[base] = tail.x; segments[base + 1u] = tail.y; segments[base + 2u] = tip.x; segments[base + 3u] = tip.y;
  segments[base + 4u] = tip.x; segments[base + 5u] = tip.y; segments[base + 6u] = wingA.x; segments[base + 7u] = wingA.y;
  segments[base + 8u] = tip.x; segments[base + 9u] = tip.y; segments[base + 10u] = wingB.x; segments[base + 11u] = wingB.y;`
  });
  return segments;
}
