// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createPolygonPublishNode} from '../../../src/gpu-spatial-analysis/rectangle-clip/rectangle-clip-kernels';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {executeGeometryNodesOnCPU} from '../polygon-triangulation/geometry-cpu-executor';

useCPUExecutorTestHooks();
it('W2-07 publishes overflow from each clipping stage even if the final total fits', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  // Logical offsets must survive the direct publication writes.
  const backing = createTransientView(graph, 'stages', 'uint32', 70);
  const stageFlags = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 260,
    length: 5
  });
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  const vertexCount = createTransientView(graph, 'count', 'uint32', 1);
  const requiredVertexCount = createTransientView(graph, 'required', 'uint32', 1);
  const node = createPolygonPublishNode(graph, {
    id: 'publish',
    operation: 'test',
    capacity: 8,
    stageFlags,
    overflow,
    vertexCount,
    requiredVertexCount,
    requiredVertexCountIndex: 4
  });
  try {
    const source = getGPUCommandNodeKernelSources([node])[0].source;
    expect(source).not.toContain('for (var stage');
    expect(source).toContain('const stageFlagsOffset: u32 = 1u;');
    const memory = createCPUGraphMemory();
    memory.write(backing, new Uint32Array(70).fill(999));
    for (const [stages, expected] of [
      [
        [9, 2, 3, 4, 99],
        [1, 4, 99]
      ],
      [
        [1, 9, 3, 4, 98],
        [1, 4, 98]
      ],
      [
        [1, 2, 9, 4, 97],
        [1, 4, 97]
      ],
      [
        [1, 2, 3, 9, 96],
        [1, 8, 96]
      ],
      [
        [1, 2, 3, 4, 95],
        [0, 4, 95]
      ]
    ] as const) {
      memory.write(stageFlags, Uint32Array.from(stages));
      executeGeometryNodesOnCPU([node], memory);
      expect([
        memory.readUint32(overflow)[0],
        memory.readUint32(vertexCount)[0],
        memory.readUint32(requiredVertexCount)[0]
      ]).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});
