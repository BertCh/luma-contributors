// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Drawing helper for operations that write edges as pairs of vertex ids plus a validity flag (hull
// boundaries, triangulation edges). The kernel resolves ids to positions on the GPU, so the
// picture shows exactly the edges the operation marked valid.

import type {RunBuffer, RunBuilder} from '../../../define-run';
import {addKernelPass} from './kernel';

/** Drawable form of an edge list. */
export type EdgeDrawing = {
  /** One (x0, y0, x1, y1) per edge, off-screen when the edge is invalid. */
  segments: RunBuffer<'float32x4'>;
  /** Two nodes per edge (start, end), for the `segments` layer. */
  nodes: RunBuffer<'float32x2'>;
  /** Edge `i` connects nodes `2i` and `2i + 1`. */
  nodeEdges: RunBuffer<'uint32x2'>;
  /** Group index per node (edge row divided by `groupSize`), for categorical colors. */
  groups: RunBuffer<'uint32'>;
};

/**
 * Adds a compute pass that turns `edges` (vertex id pairs into `positions`) into drawable segments.
 * Edge `i` is drawn when `validity[i]` is nonzero. `groupSize` edges form one group.
 */
export function addEdgeDrawing(
  builder: RunBuilder,
  id: string,
  positions: RunBuffer<'float32x2'>,
  edges: RunBuffer<'uint32x2'>,
  validity: RunBuffer<'uint32'>,
  groupSize = edges.length
): EdgeDrawing {
  const segments = builder.allocate(`${id}-segments`, 'float32x4', edges.length);
  const nodes = builder.allocate(`${id}-nodes`, 'float32x2', edges.length * 2);
  const groups = builder.allocate(`${id}-groups`, 'uint32', edges.length * 2);
  const nodeEdges = builder.upload(
    `${id}-node-edges`,
    'uint32x2',
    Uint32Array.from({length: edges.length * 2}, (_, slot) => Math.floor(slot / 2) * 2 + (slot % 2))
  );
  addKernelPass(builder, {
    id: `${id}-edge-pass`,
    invocationCount: edges.length,
    bindings: [
      {name: 'positions', buffer: positions, access: 'read'},
      {name: 'edges', buffer: edges, access: 'read'},
      {name: 'validity', buffer: validity, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'},
      {name: 'nodes', buffer: nodes, access: 'read_write'},
      {name: 'groups', buffer: groups, access: 'read_write'}
    ],
    body: /* wgsl */ `
  var start = vec2f(-9.0);
  var end = vec2f(-9.0);
  if (validity[validityOffset + index] != 0u) {
    let edge = edges[edgesOffset + index];
    start = positions[positionsOffset + edge.x];
    end = positions[positionsOffset + edge.y];
  }
  segments[segmentsOffset + index] = vec4f(start, end);
  nodes[nodesOffset + index * 2u] = start;
  nodes[nodesOffset + index * 2u + 1u] = end;
  let group = index / ${Math.max(groupSize, 1)}u;
  groups[groupsOffset + index * 2u] = group;
  groups[groupsOffset + index * 2u + 1u] = group;`
  });
  return {segments, nodes, nodeEdges, groups};
}
