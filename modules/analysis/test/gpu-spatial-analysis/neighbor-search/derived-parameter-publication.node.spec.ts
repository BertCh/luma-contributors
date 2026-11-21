// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createNeighborSearchDerivedParameterNodes} from '../../../src/gpu-spatial-analysis/neighbor-search/neighbor-search-derived-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {executeGeometryNodesOnCPU} from '../polygon-triangulation/geometry-cpu-executor';

useCPUExecutorTestHooks();
for (const geographic of [false, true]) {
  it(`W2-06 publishes four extent words and preserves temporal parameters (${geographic ? 'geographic' : 'planar'})`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const createView = vi.spyOn(graph, 'createDataView');
    try {
      const result = createNeighborSearchDerivedParameterNodes(graph, {
        id: 'derived',
        operation: 'test',
        radius: 7,
        timeRadius: 3,
        geographic,
        positions: createTransientView(graph, 'positions', 'float32x2', 2),
        times: createTransientView(graph, 'times', 'float32', 2)
      });
      const publisher = result.nodes.at(-1)!;
      const source = getGPUCommandNodeKernelSources([publisher])[0].source;
      expect(source).not.toContain('for (var word');
      const memory = createCPUGraphMemory();
      for (const [name, values] of [
        ['point-bounds', [10, 20, 30, 40]],
        ['time-bounds', [-2, 0, 50, 0]]
      ] as const) {
        const view = createView.mock.results.find(
          call => call.type === 'return' && call.value.buffer.id === `derived-${name}`
        )?.value;
        if (view) memory.write(view, Float32Array.from(values));
      }
      executeGeometryNodesOnCPU([publisher], memory);
      expect(Array.from(memory.readFloat32(result.parameters))).toEqual([
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
      if (!geographic) {
        for (const word of ['', ' + 1u', ' + 2u', ' + 3u'])
          expect(source).toContain(`bounds[boundsOffset${word}]`);
      }
    } finally {
      createView.mockRestore();
      device.destroy();
    }
  });
}

it('B5 keeps poison inside the position and time bindings before logical input data', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positionBacking = createTransientView(graph, 'position-backing', 'float32x2', 35);
  const timeBacking = createTransientView(graph, 'time-backing', 'float32', 67);
  const positions = graph.createDataView(positionBacking.buffer, {
    format: 'float32x2',
    byteOffset: 264,
    length: 2
  });
  const times = graph.createDataView(timeBacking.buffer, {
    format: 'float32',
    byteOffset: 260,
    length: 2
  });
  const result = createNeighborSearchDerivedParameterNodes(graph, {
    id: 'offset-derived',
    operation: 'test',
    positions,
    times,
    radius: 7,
    timeRadius: 3
  });
  try {
    const sources = getGPUCommandNodeKernelSources(result.nodes)
      .map(kernel => kernel.source)
      .join('\n');
    expect(sources).toContain('const positionsOffset: u32 = 2u;');
    expect(sources).toContain('const timesOffset: u32 = 1u;');
    const memory = createCPUGraphMemory();
    memory.write(positionBacking, new Float32Array(70).fill(-999));
    memory.write(timeBacking, new Float32Array(67).fill(999));
    memory.write(positions, Float32Array.of(10, 20, 30, 40));
    memory.write(times, Float32Array.of(-2, 50));
    executeGeometryNodesOnCPU(result.nodes, memory);
    expect(Array.from(memory.readFloat32(result.parameters))).toEqual([
      10, 20, 30, 40, 7, 0, 1, 0, 1, 0, 0, 0, 3, -2, 50
    ]);
  } finally {
    device.destroy();
  }
});
