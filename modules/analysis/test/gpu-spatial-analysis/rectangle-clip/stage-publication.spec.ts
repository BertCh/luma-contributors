// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createPolygonPublishNode} from '../../../src/gpu-spatial-analysis/rectangle-clip/rectangle-clip-kernels';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('W2-07 keeps early-stage overflow when the final stage fits', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const backing = new Uint32Array(70).fill(999);
  backing.set([9, 2, 3, 4, 99], 65);
  const stageBuffer = createInputBuffer(device, backing);
  const overflowBuffer = createOutputBuffer(device, 1);
  const countBuffer = createOutputBuffer(device, 1);
  const requiredBuffer = createOutputBuffer(device, 1);
  const stageBacking = importGraphBuffer(graph, 'stage-backing', stageBuffer, 'uint32', 70);
  graph.add(
    createPolygonPublishNode(graph, {
      id: 'publish',
      operation: 'test',
      capacity: 8,
      stageFlags: graph.createDataView(stageBacking.buffer, {
        format: 'uint32',
        byteOffset: 260,
        length: 5
      }),
      overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1),
      vertexCount: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
      requiredVertexCount: importGraphBuffer(graph, 'required', requiredBuffer, 'uint32', 1),
      requiredVertexCountIndex: 4
    })
  );
  const compiled = graph.compile();
  try {
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
      backing.set(stages, 65);
      stageBuffer.write(backing);
      submitGraph(device, compiled, undefined);
      expect([
        (await readUint32(overflowBuffer, 1))[0],
        (await readUint32(countBuffer, 1))[0],
        (await readUint32(requiredBuffer, 1))[0]
      ]).toEqual(expected);
    }
  } finally {
    compiled.destroy();
    stageBuffer.destroy();
    overflowBuffer.destroy();
    countBuffer.destroy();
    requiredBuffer.destroy();
  }
});
