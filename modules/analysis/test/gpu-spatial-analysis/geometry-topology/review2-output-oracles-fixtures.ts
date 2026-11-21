// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUOffsetCurveComponents} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components';
import {createPolygonizeDanglePruningNodes} from '../../../src/gpu-spatial-analysis/geometry-topology/polygonize-dangle-pruning';
import type {OracleAllocate} from './noding-polygonize-output-oracles-fixtures';

export const REVIEW2_OFFSET_ORIGINAL = [0, 0, 2, 0, 4, 0, 6, 0];
export const REVIEW2_OFFSET_RAW = [0, 1, 2, 1, 0, 1, 2, 1];
export const REVIEW2_OFFSET_OPTIONS = {
  distance: 1,
  tolerance: 0,
  rowsPerVertex: 1,
  intersectionCapacity: 8,
  segmentCapacity: 16,
  vertexCapacity: 2,
  componentCapacity: 1
};

export function createReview2OffsetCase(graph: GPUCommandGraph, allocate: OracleAllocate) {
  const original = {
    positions: allocate('review2-original', 'float32x2', 4, REVIEW2_OFFSET_ORIGINAL),
    lineOffsets: allocate('review2-original-offsets', 'uint32', 2, [0, 4])
  };
  const raw = {
    positions: allocate('review2-raw', 'float32x2', 4, REVIEW2_OFFSET_RAW),
    lineOffsets: allocate('review2-raw-offsets', 'uint32', 2, [0, 4])
  };
  const output = {
    positions: allocate('review2-offset-positions', 'float32x2', 2),
    components: allocate('review2-offset-components', 'uint32x4', 1),
    sourceEdges: allocate('review2-offset-source-edges', 'uint32', 2),
    status: allocate('review2-offset-status', 'uint32x4', 2)
  };
  // By traversal: forward raw row 0 cancels reverse row 1. The surviving
  // forward raw row 2 joins (0,1) to (2,1), so both output vertices own row 2.
  return {
    output,
    expected: {
      positions: [0, 1, 2, 1],
      components: [0, 2, 0, 0],
      sourceEdges: [2, 2],
      status: [1, 2, 1, 2, 0, 0, 0, 0]
    },
    nodes: new GPUOffsetCurveComponents({
      ...REVIEW2_OFFSET_OPTIONS,
      original,
      raw,
      output
    }).getCommandNodes(graph)
  };
}

export function createReview2StackedSquaresCase(
  graph: GPUCommandGraph,
  allocate: OracleAllocate,
  squareCount: number
) {
  const values: number[] = [];
  for (let square = 0; square < squareCount; square++) {
    const corners = [
      [0, 8 * square],
      [1, 8 * square],
      [1, 8 * square + 1],
      [0, 8 * square + 1]
    ];
    for (let side = 0; side < 4; side++) {
      const first = corners[side]!;
      const last = corners[(side + 1) % 4]!;
      values.push(...first, ...last, ...last, ...first);
    }
  }
  const endpointCount = 8 * squareCount;
  const endpoints = allocate('review2-stacked-endpoints', 'float32x4', endpointCount, values);
  const groups = allocate(
    'review2-stacked-groups',
    'uint32',
    endpointCount,
    new Array(endpointCount).fill(0)
  );
  const count = allocate('review2-stacked-count', 'uint32', 1, [endpointCount]);
  const candidateCounts = allocate('review2-stacked-visits', 'uint32', endpointCount);
  // Every corner occurs twice as a directed start. Other corners are >=1 away,
  // much farther than the 1/64 cells; distant squares are separated by 7 units.
  // Thus each endpoint inspects exactly its two coincident starts, at any size.
  const {nodes} = createPolygonizeDanglePruningNodes(graph, {
    id: 'review2-stacked',
    endpoints,
    groups,
    count,
    vertexTolerance: 1 / 64,
    candidateCounts
  });
  return {
    output: {candidateCounts},
    expected: {candidateCounts: new Array<number>(endpointCount).fill(2)},
    nodes
  };
}
