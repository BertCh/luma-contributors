// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSegmentIntersection} from '../../../src/gpu-spatial-analysis/segment-intersection/gpu-segment-intersection';
import type {MultipartTestAllocate} from '../geometry-topology/multipart-line-forms-fixtures';

export function createMultipartSegmentCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate
) {
  const lines = (prefix: string) => ({
    kind: 'lines' as const,
    positions: allocate(`${prefix}-positions`, 'float32x2', 4, [0, 0, 2, 2, 0, 2, 2, 0]),
    lineOffsets: allocate(`${prefix}-parts`, 'uint32', 3, [0, 2, 4]),
    featureOffsets: allocate(`${prefix}-input-features`, 'uint32', 3, [0, 0, 2])
  });
  const output = {
    pairs: {
      leftRows: allocate('left-rows', 'uint32', 4),
      rightRows: allocate('right-rows', 'uint32', 4),
      status: {
        count: allocate('count', 'uint32', 1),
        requiredCount: allocate('required', 'uint32', 1),
        overflow: allocate('overflow', 'uint32', 1),
        candidateOverflow: allocate('candidate-overflow', 'uint32', 1)
      }
    },
    leftFeatures: allocate('left-features', 'uint32', 4),
    rightFeatures: allocate('right-features', 'uint32', 4),
    leftRings: allocate('left-rings', 'uint32', 4),
    rightRings: allocate('right-rings', 'uint32', 4),
    kinds: allocate('kinds', 'uint32', 4),
    points: allocate('points', 'float32x2', 4),
    endPoints: allocate('end-points', 'float32x2', 4),
    status: {uncertainCount: allocate('uncertain', 'uint32', 1)}
  };
  // Two identical crossed-part features: each same-part pair overlaps, each opposite pair
  // crosses at (1,1). Feature row 0 is empty; all segments belong to feature row 1.
  // Independent Shapely intersections: join-construction-review.oracle.py.
  const expected = {
    'pairs.leftRows': [0, 0, 2, 2],
    'pairs.rightRows': [0, 2, 0, 2],
    'pairs.status.count': [4],
    'pairs.status.requiredCount': [4],
    'pairs.status.overflow': [0],
    'pairs.status.candidateOverflow': [0],
    leftFeatures: [1, 1, 1, 1],
    rightFeatures: [1, 1, 1, 1],
    leftRings: [0, 0, 1, 1],
    rightRings: [0, 1, 0, 1],
    kinds: [4, 1, 1, 4],
    points: [0, 0, 1, 1, 1, 1, 0, 2],
    endPoints: [2, 2, 1, 1, 1, 1, 2, 0],
    'status.uncertainCount': [0]
  };
  return {
    output,
    expected,
    nodes: new GPUSegmentIntersection({
      left: lines('left'),
      right: lines('right'),
      output
    }).getCommandNodes(graph)
  };
}
