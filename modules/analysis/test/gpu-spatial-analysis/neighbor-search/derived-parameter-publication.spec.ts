// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNeighborSearchDerivedParameterNodes} from '../../../src/gpu-spatial-analysis/neighbor-search/neighbor-search-derived-parameters';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

for (const geographic of [false, true]) {
  it(`W2-06 publishes derived extent and temporal words with offset inputs (${geographic ? 'geographic' : 'planar'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const positionData = new Float32Array(70).fill(-999);
    positionData.set([10, 20, 30, 40], 66);
    const timeData = new Float32Array(67).fill(999);
    timeData.set([-2, 50], 65);
    const positionBuffer = createInputBuffer(device, positionData);
    const timeBuffer = createInputBuffer(device, timeData);
    const outputBuffer = createOutputBuffer(device, 15);
    const positionsBacking = importGraphBuffer(
      graph,
      'position-backing',
      positionBuffer,
      'float32x2',
      35
    );
    const timesBacking = importGraphBuffer(graph, 'time-backing', timeBuffer, 'float32', 67);
    const result = createNeighborSearchDerivedParameterNodes(graph, {
      id: 'derived',
      operation: 'test',
      geographic,
      radius: 7,
      timeRadius: 3,
      positions: graph.createDataView(positionsBacking.buffer, {
        format: 'float32x2',
        byteOffset: 264,
        length: 2
      }),
      times: graph.createDataView(timesBacking.buffer, {
        format: 'float32',
        byteOffset: 260,
        length: 2
      })
    });
    for (const node of result.nodes) graph.add(node);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'read-parameters',
        operation: 'test',
        invocationCount: 15,
        bindings: [
          {name: 'parameters', view: result.parameters, type: 'f32', access: 'read'},
          {
            name: 'output',
            view: importGraphBuffer(graph, 'output', outputBuffer, 'float32', 15),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: 'output[outputOffset + index] = parameters[parametersOffset + index];'
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(outputBuffer, 15)).toEqual([
        ...(geographic ? [-180, -90, 180, 90] : [10, 20, 30, 40]),
        7,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        3,
        -2,
        50
      ]);
    } finally {
      compiled.destroy();
      positionBuffer.destroy();
      timeBuffer.destroy();
      outputBuffer.destroy();
    }
  });
}
