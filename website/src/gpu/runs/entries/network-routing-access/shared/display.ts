// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Display adapters: tiny GPU kernels that reshape an operation's output buffers into the
// segment lists the run layers draw (for example padded vertex ids into line segments). They read
// the operation's output on the GPU; nothing is read back or recomputed on the CPU.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

/** Segment endpoint written where a padded slot has no vertex: far outside every view. */
const OFFSCREEN = '-100.0';

/**
 * Turns `routeIndex`-th slice (`stride` ids, padded with 0xffffffff) of a vertex-id buffer into
 * (x0, y0, x1, y1) segments between consecutive vertices. `nodePositions` is a flat x, y array.
 */
export function addVertexSegments(
  builder: RunBuilder,
  options: {
    id: string;
    vertices: RunBuffer<'uint32'>;
    nodePositions: RunBuffer<'float32'>;
    routeIndex: number;
    stride: number;
  }
): RunBuffer<'float32'> {
  const segmentCount = options.stride - 1;
  const segments = builder.allocate(`${options.id}-segments`, 'float32', segmentCount * 4);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-expand`,
      operation: 'display-vertex-segments',
      invocationCount: segmentCount,
      bindings: [
        {name: 'vertices', view: options.vertices.view, type: 'u32', access: 'read'},
        {name: 'nodePositions', view: options.nodePositions.view, type: 'f32', access: 'read'},
        {name: 'segments', view: segments.view, type: 'f32', access: 'read_write'}
      ],
      body: `
  let first = vertices[verticesOffset + ${options.routeIndex * options.stride}u + index];
  let second = vertices[verticesOffset + ${options.routeIndex * options.stride}u + index + 1u];
  let isValid = first != 0xffffffffu && second != 0xffffffffu;
  let slot = segmentsOffset + index * 4u;
  segments[slot] = select(${OFFSCREEN}, nodePositions[nodePositionsOffset + first * 2u], isValid);
  segments[slot + 1u] = select(${OFFSCREEN}, nodePositions[nodePositionsOffset + first * 2u + 1u], isValid);
  segments[slot + 2u] = select(${OFFSCREEN}, nodePositions[nodePositionsOffset + second * 2u], isValid);
  segments[slot + 3u] = select(${OFFSCREEN}, nodePositions[nodePositionsOffset + second * 2u + 1u], isValid);`
    })
  );
  return segments;
}

/**
 * Per-slot values drawn on slot lines: copies `source` (one float32 or uint32 value per slot) into
 * both halves of a `2 * count` float32 buffer, because a slot line reads its value at its first
 * and second endpoint.
 */
export function addSlotValues(
  builder: RunBuilder,
  options: {id: string; source: RunBuffer<'float32'> | RunBuffer<'uint32'>; count: number}
): RunBuffer<'float32'> {
  const doubled = builder.allocate(`${options.id}-doubled`, 'float32', options.count * 2);
  const isInteger = options.source.format === 'uint32';
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-double`,
      operation: 'display-slot-values',
      invocationCount: options.count,
      bindings: [
        {
          name: 'source',
          view: options.source.view,
          type: isInteger ? 'u32' : 'f32',
          access: 'read'
        },
        {name: 'doubled', view: doubled.view, type: 'f32', access: 'read_write'}
      ],
      body: `
  let value = ${isInteger ? 'f32(source[sourceOffset + index])' : 'source[sourceOffset + index]'};
  doubled[doubledOffset + index] = value;
  doubled[doubledOffset + ${options.count}u + index] = value;`
    })
  );
  return doubled;
}

/**
 * Interleaves two parallel uint32 columns (`from`, `to`) into (from, to) rows, the edge-pair layout
 * of the segments layer. Padding rows (0xffffffff) point at `fallbackNode`, a node kept offscreen.
 */
export function addEdgePairs(
  builder: RunBuilder,
  options: {
    id: string;
    from: RunBuffer<'uint32'>;
    to: RunBuffer<'uint32'>;
    fallbackNode: number;
  }
): RunBuffer<'uint32'> {
  const count = options.from.length;
  const pairs = builder.allocate(`${options.id}-pairs`, 'uint32', count * 2);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-interleave`,
      operation: 'display-edge-pairs',
      invocationCount: count,
      bindings: [
        {name: 'sources', view: options.from.view, type: 'u32', access: 'read'},
        {name: 'targets', view: options.to.view, type: 'u32', access: 'read'},
        {name: 'pairs', view: pairs.view, type: 'u32', access: 'read_write'}
      ],
      body: `
  let first = sources[sourcesOffset + index];
  let second = targets[targetsOffset + index];
  let isValid = first != 0xffffffffu && second != 0xffffffffu;
  pairs[pairsOffset + index * 2u] = select(${options.fallbackNode}u, first, isValid);
  pairs[pairsOffset + index * 2u + 1u] = select(${options.fallbackNode}u, second, isValid);`
    })
  );
  return pairs;
}

/** Copies `length` values (float32 or uint32) starting at `offset` of `source` into a float32 buffer. */
export function addSlice(
  builder: RunBuilder,
  options: {
    id: string;
    source: RunBuffer<'float32'> | RunBuffer<'uint32'>;
    offset: number;
    length: number;
  }
): RunBuffer<'float32'> {
  const slice = builder.allocate(`${options.id}-slice`, 'float32', options.length);
  const isInteger = options.source.format === 'uint32';
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-copy`,
      operation: 'display-slice',
      invocationCount: options.length,
      bindings: [
        {
          name: 'source',
          view: options.source.view,
          type: isInteger ? 'u32' : 'f32',
          access: 'read'
        },
        {name: 'slice', view: slice.view, type: 'f32', access: 'read_write'}
      ],
      body: `slice[sliceOffset + index] = ${isInteger ? 'f32(source[sourceOffset + ' + options.offset + 'u + index])' : 'source[sourceOffset + ' + options.offset + 'u + index]'};`
    })
  );
  return slice;
}

/**
 * Copies a graph-owned uint32 view (an operation or recipe output with no buffer of its own) into a
 * run buffer that the layers can draw.
 */
export function addUint32Copy(
  builder: RunBuilder,
  options: {
    id: string;
    view: GraphDataView<'uint32'>;
    /** WGSL statements turning `value` into the stored `result` (default: copy). */
    transform?: string;
  }
): RunBuffer<'uint32'> {
  const copy = builder.allocate(`${options.id}-copy`, 'uint32', options.view.length);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-copy`,
      operation: 'display-copy',
      invocationCount: options.view.length,
      bindings: [
        {name: 'source', view: options.view, type: 'u32', access: 'read'},
        {name: 'copy', view: copy.view, type: 'u32', access: 'read_write'}
      ],
      body: `let value = source[sourceOffset + index];
  var result = value;
  ${options.transform ?? ''}
  copy[copyOffset + index] = result;`
    })
  );
  return copy;
}

/**
 * Facility colors: row `index` holds `index` where `flags[index]` equals `flagValue` and the invalid
 * value 0xffffffff otherwise, so selected (or open) sites take their categorical color and the
 * others draw muted.
 */
export function addFlaggedIds(
  builder: RunBuilder,
  options: {id: string; flags: RunBuffer<'uint32'>; count: number; flagValue?: number}
): RunBuffer<'uint32'> {
  const ids = builder.allocate(`${options.id}-ids`, 'uint32', options.count);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-flagged`,
      operation: 'display-flagged-ids',
      invocationCount: options.count,
      bindings: [
        {name: 'flags', view: options.flags.view, type: 'u32', access: 'read'},
        {name: 'ids', view: ids.view, type: 'u32', access: 'read_write'}
      ],
      body: `ids[idsOffset + index] = select(0xffffffffu, index, flags[flagsOffset + index] == ${options.flagValue ?? 1}u);`
    })
  );
  return ids;
}

/** (x0, y0, x1, y1) segments joining two parallel float32x2 position buffers row by row. */
export function addPairSegments(
  builder: RunBuilder,
  options: {id: string; from: RunBuffer<'float32x2'>; to: RunBuffer<'float32x2'>}
): RunBuffer<'float32x4'> {
  const segments = builder.allocate(`${options.id}-pairs`, 'float32x4', options.from.length);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-join`,
      operation: 'display-pair-segments',
      invocationCount: options.from.length,
      bindings: [
        {name: 'origins', view: options.from.view, type: 'f32', access: 'read'},
        {name: 'targets', view: options.to.view, type: 'f32', access: 'read'},
        {name: 'segments', view: segments.view, type: 'f32', access: 'read_write'}
      ],
      body: `
  let slot = segmentsOffset + index * 4u;
  segments[slot] = origins[originsOffset + index * 2u];
  segments[slot + 1u] = origins[originsOffset + index * 2u + 1u];
  segments[slot + 2u] = targets[targetsOffset + index * 2u];
  segments[slot + 3u] = targets[targetsOffset + index * 2u + 1u];`
    })
  );
  return segments;
}

/**
 * Facility id per demand location, or the invalid value 0xffffffff where the nearest facility is
 * farther than `threshold` (a one-row float32 buffer the run rewrites each frame).
 */
export function addThresholdedIds(
  builder: RunBuilder,
  options: {
    id: string;
    facilities: RunBuffer<'uint32'>;
    costs: RunBuffer<'float32'>;
    threshold: RunBuffer<'float32'>;
  }
): RunBuffer<'uint32'> {
  const ids = builder.allocate(`${options.id}-ids`, 'uint32', options.facilities.length);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-thresholded`,
      operation: 'display-thresholded-ids',
      invocationCount: options.facilities.length,
      bindings: [
        {name: 'facilities', view: options.facilities.view, type: 'u32', access: 'read'},
        {name: 'costs', view: options.costs.view, type: 'f32', access: 'read'},
        {name: 'threshold', view: options.threshold.view, type: 'f32', access: 'read'},
        {name: 'ids', view: ids.view, type: 'u32', access: 'read_write'}
      ],
      body: `
  let isCovered = costs[costsOffset + index] <= threshold[thresholdOffset];
  ids[idsOffset + index] = select(0xffffffffu, facilities[facilitiesOffset + index], isCovered);`
    })
  );
  return ids;
}

/**
 * One value per node read out of an operation's cost matrix, as a footprint for the segments
 * layer: 1 at cost 0 falling to `floor` at `limit`, and 0 where the cost is above `limit` or
 * unreached. `row` mode reads the row `selector[0]` of a start-by-node matrix (a facility's cost to
 * every node); `column` mode reads column `selector[0]` of a node-by-event matrix (every node's
 * cost to one event). `limit` is a one-row float32 buffer the run may rewrite each frame.
 */
export function addMatrixFootprint(
  builder: RunBuilder,
  options: {
    id: string;
    costs: RunBuffer<'float32'>;
    nodeCount: number;
    /** Columns of the matrix: `nodeCount` in row mode, the event count in column mode. */
    columnCount: number;
    mode: 'row' | 'column';
    selector: RunBuffer<'uint32'>;
    limit: RunBuffer<'float32'>;
    floor?: number;
  }
): RunBuffer<'float32'> {
  const footprint = builder.allocate(`${options.id}-footprint`, 'float32', options.nodeCount);
  const cell =
    options.mode === 'row'
      ? `selector[selectorOffset] * ${options.columnCount}u + index`
      : `index * ${options.columnCount}u + selector[selectorOffset]`;
  const floor = options.floor ?? 0.3;
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-footprint`,
      operation: 'display-matrix-footprint',
      invocationCount: options.nodeCount,
      bindings: [
        {name: 'costs', view: options.costs.view, type: 'f32', access: 'read'},
        {name: 'selector', view: options.selector.view, type: 'u32', access: 'read'},
        {name: 'limit', view: options.limit.view, type: 'f32', access: 'read'},
        {name: 'footprint', view: footprint.view, type: 'f32', access: 'read_write'}
      ],
      body: `
  let cost = costs[costsOffset + ${cell}];
  let reach = limit[limitOffset];
  footprint[footprintOffset + index] = select(0.0, 1.0 - (1.0 - ${floor.toFixed(3)}) * cost / reach, cost >= 0.0 && cost <= reach);`
    })
  );
  return footprint;
}

/**
 * (x0, y0, x1, y1) segments joining consecutive rows of a float32x2 buffer inside each track of
 * `trackLength` rows. The slot that would join two tracks, or a row with `isShown` false, is
 * parked far outside the view. `flags` (optional, uint32 per row) keeps a segment only where the
 * flag of its second row is one of `flagValues`.
 */
export function addConsecutiveSegments(
  builder: RunBuilder,
  options: {
    id: string;
    positions: RunBuffer<'float32x2'>;
    trackLength: number;
    flags?: RunBuffer<'uint32'>;
    flagValues?: readonly number[];
  }
): RunBuffer<'float32x4'> {
  const count = options.positions.length;
  const segments = builder.allocate(`${options.id}-legs`, 'float32x4', count);
  const keep = options.flags
    ? options.flagValues?.map(value => `flags[flagsOffset + index] == ${value}u`).join(' || ') ||
      'flags[flagsOffset + index] != 0u'
    : 'true';
  const bindings = [
    {name: 'points', view: options.positions.view, type: 'f32' as const, access: 'read' as const},
    ...(options.flags
      ? [{name: 'flags', view: options.flags.view, type: 'u32' as const, access: 'read' as const}]
      : []),
    {name: 'segments', view: segments.view, type: 'f32' as const, access: 'read_write' as const}
  ];
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-legs`,
      operation: 'display-consecutive-segments',
      invocationCount: count,
      bindings,
      body: `
  let isValid = index % ${options.trackLength}u != 0u && (${keep});
  let slot = segmentsOffset + index * 4u;
  let previous = select(index, index - 1u, index > 0u);
  segments[slot] = select(${OFFSCREEN}, points[pointsOffset + previous * 2u], isValid);
  segments[slot + 1u] = select(${OFFSCREEN}, points[pointsOffset + previous * 2u + 1u], isValid);
  segments[slot + 2u] = select(${OFFSCREEN}, points[pointsOffset + index * 2u], isValid);
  segments[slot + 3u] = select(${OFFSCREEN}, points[pointsOffset + index * 2u + 1u], isValid);`
    })
  );
  return segments;
}

/**
 * (x0, y0, x1, y1) segments for the roads whose per-node `values` are above `threshold` at either
 * end, parked far outside the view for every other road. A solid-color lines layer then lights a
 * zone of the network, which the segments layer cannot do (its faded-out edges draw opaque).
 */
export function addZoneSegments(
  builder: RunBuilder,
  options: {
    id: string;
    nodes: RunBuffer<'float32x2'>;
    edges: RunBuffer<'uint32x2'>;
    values: RunBuffer<'float32'>;
    threshold?: number;
  }
): RunBuffer<'float32x4'> {
  const segments = builder.allocate(`${options.id}-zone`, 'float32x4', options.edges.length);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-zone`,
      operation: 'display-zone-segments',
      invocationCount: options.edges.length,
      bindings: [
        {name: 'nodes', view: options.nodes.view, type: 'f32', access: 'read'},
        {name: 'edges', view: options.edges.view, type: 'u32', access: 'read'},
        {name: 'values', view: options.values.view, type: 'f32', access: 'read'},
        {name: 'segments', view: segments.view, type: 'f32', access: 'read_write'}
      ],
      body: `
  let first = edges[edgesOffset + index * 2u];
  let second = edges[edgesOffset + index * 2u + 1u];
  let isInside = values[valuesOffset + first] > ${(options.threshold ?? 0).toFixed(3)} && values[valuesOffset + second] > ${(options.threshold ?? 0).toFixed(3)};
  let slot = segmentsOffset + index * 4u;
  segments[slot] = select(${OFFSCREEN}, nodes[nodesOffset + first * 2u], isInside);
  segments[slot + 1u] = select(${OFFSCREEN}, nodes[nodesOffset + first * 2u + 1u], isInside);
  segments[slot + 2u] = select(${OFFSCREEN}, nodes[nodesOffset + second * 2u], isInside);
  segments[slot + 3u] = select(${OFFSCREEN}, nodes[nodesOffset + second * 2u + 1u], isInside);`
    })
  );
  return segments;
}
