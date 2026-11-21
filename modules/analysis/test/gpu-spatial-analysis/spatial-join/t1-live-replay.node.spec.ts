// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {adaptUniforms} from './multipart-point-joins-executor';

useCPUExecutorTestHooks();
it('t1 live anti replay preserves an empty feature after shrinking every terminal and clearing upstream loss', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device),
    memory = createCPUGraphMemory();
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const features = createTransientView(graph, 'features', 'uint32', 4),
    polygons = createTransientView(graph, 'polygons', 'uint32', 4),
    rings = createTransientView(graph, 'rings', 'uint32', 4);
  const positions = createTransientView(graph, 'positions', 'float32x2', 12),
    point = createTransientView(graph, 'point', 'float32x2', 1);
  const live = scalar('live'),
    loss = scalar('loss'),
    count = scalar('count'),
    candidateOverflow = scalar('candidateOverflow'),
    uncertainCount = scalar('uncertainCount');
  const ids = createTransientView(graph, 'ids', 'uint32', 3),
    truncated = createTransientView(graph, 'truncated', 'uint32', 3);
  memory.write(features, Uint32Array.of(0, 1, 2, 3));
  memory.write(polygons, Uint32Array.of(0, 1, 2, 3));
  memory.write(rings, Uint32Array.of(0, 4, 8, 12));
  memory.write(
    positions,
    Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2, 4, 0, 6, 0, 6, 2, 4, 2, 8, 0, 10, 0, 10, 2, 8, 2)
  );
  memory.write(point, Float32Array.of(1, 1));
  const join = new GPUSpatialPredicateJoin({
    left: {
      kind: 'polygons',
      positions,
      featureOffsets: features,
      polygonOffsets: polygons,
      ringOffsets: rings
    },
    right: {kind: 'points', positions: point},
    leftFeatureCount: live,
    leftStatus: {overflow: loss},
    predicate: 'intersects',
    how: 'anti',
    output: {
      unmatched: {ids, status: {count, overflow: scalar('overflow'), candidateOverflow}},
      candidateTruncated: truncated,
      status: {uncertainCount}
    }
  });
  const nodes = adaptUniforms(join.getCommandNodes(graph));
  try {
    for (const [liveCount, upstreamLoss, expected] of [
      [3, 0, [1, 2]],
      [1, 0, []],
      [2, 0, [1]],
      [2, 1, []],
      [2, 0, [1]]
    ] as const) {
      if (liveCount === 1) {
        memory.write(features, Uint32Array.of(0, 1, 1, 1));
        memory.write(polygons, Uint32Array.of(0, 1, 1, 1));
        memory.write(rings, Uint32Array.of(0, 4, 4, 4));
      }
      memory.write(live, Uint32Array.of(liveCount));
      memory.write(loss, Uint32Array.of(upstreamLoss));
      executeCommandNodesOnCPU(nodes, memory);
      const actual = {
        count: memory.readUint32(count)[0],
        candidateOverflow: memory.readUint32(candidateOverflow)[0],
        uncertain: memory.readUint32(uncertainCount)[0],
        truncated: Array.from(memory.readUint32(truncated))
      };
      expect(actual, `live=${liveCount},loss=${upstreamLoss}`).toEqual({
        count: expected.length,
        candidateOverflow: upstreamLoss,
        uncertain: 0,
        truncated: upstreamLoss ? [1, 1, 0] : [0, 0, 0]
      });
      expect(Array.from(memory.readUint32(ids)).slice(0, expected.length)).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});
