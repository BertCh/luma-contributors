// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createEncounterScanNode,
  createEncounterOverflowNode
} from '../../../src/gpu-spatial-analysis/trajectory-encounters/encounter-kernels';

/** Three coincident tracks produce exactly three unordered hits; seed the count at uint32-1. */
export function createTrajectoryCounterOverflowFixture(
  graph: GPUCommandGraph,
  output?: {
    state: GraphDataView<'uint32'>;
    pairs: GraphDataView<'uint32x2'>;
    overflow: GraphDataView<'uint32'>;
  }
) {
  const positions = createTransientView(graph, 'positions', 'float32x3', 3);
  const cellOffsets = createTransientView(graph, 'cells', 'uint32', 2);
  const objectIds = createTransientView(graph, 'objects', 'uint32', 3);
  const distance = createTransientView(graph, 'distance', 'float32', 1);
  const state = output?.state ?? createTransientView(graph, 'state', 'uint32', 1);
  const hitPairs = output?.pairs ?? createTransientView(graph, 'pairs', 'uint32x2', 2);
  const hitBuckets = createTransientView(graph, 'buckets', 'uint32', 2);
  const hitDistances = createTransientView(graph, 'distances', 'float32', 2);
  const overflow = output?.overflow ?? createTransientView(graph, 'overflow', 'uint32', 1);
  const shape = {
    trackCount: 3,
    bucketCount: 1,
    hitCapacity: 2,
    pairCapacity: 2,
    columns: 1,
    rows: 1,
    bounds: [-1, -1, 1, 1] as const,
    cellSize: 2
  };
  const seed = createWGSLKernelNode(graph, {
    id: 'seed',
    operation: 'TrajectoryOverflowFixture',
    invocationCount: 1,
    bindings: [
      {name: 'positions', view: positions, type: 'f32', access: 'read_write'},
      {name: 'cells', view: cellOffsets, type: 'u32', access: 'read_write'},
      {name: 'objects', view: objectIds, type: 'u32', access: 'read_write'},
      {name: 'distance', view: distance, type: 'f32', access: 'read_write'},
      {name: 'state', view: state, type: 'u32', access: 'read_write'},
      {name: 'pairs', view: hitPairs, type: 'u32', access: 'read_write'},
      {name: 'buckets', view: hitBuckets, type: 'u32', access: 'read_write'},
      {name: 'distances', view: hitDistances, type: 'f32', access: 'read_write'}
    ],
    body: `for (var row = 0u; row < 3u; row++) {
      positions[row * 3u] = 0.0; positions[row * 3u + 1u] = 0.0; positions[row * 3u + 2u] = 0.5;
      objects[row] = row;
    }
    cells[0] = 0u; cells[1] = 3u; distance[0] = 1.0; state[0] = 0xfffffffeu;
    pairs[0] = 99u; pairs[1] = 99u; pairs[2] = 99u; pairs[3] = 99u;
    buckets[0] = 99u; buckets[1] = 99u; distances[0] = 99.0; distances[1] = 99.0;`
  });
  const scan = createEncounterScanNode(graph, {
    id: 'scan',
    shape,
    positions,
    cellOffsets,
    objectIds,
    distance,
    state,
    hitPairs,
    hitBuckets,
    hitDistances
  });
  const publish = createEncounterOverflowNode(graph, {
    id: 'publish-overflow',
    shape,
    state,
    flag: overflow
  });
  return {nodes: [seed, scan, publish], state, pairs: hitPairs, overflow};
}
