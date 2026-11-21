// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {executeGeometryNodesOnCPU} from '../polygon-triangulation/geometry-cpu-executor';

useCPUExecutorTestHooks();
it('W2-05 publishes exactly the four sample-cell parameters', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const createView = vi.spyOn(graph, 'createDataView');
  const preparation = new GPUStudyWindowPreparation({
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    samplingGrid: [4, 2]
  });
  try {
    const node = preparation
      .getCommandNodes(graph)
      .find(candidate => candidate.id.endsWith('-sample-parameters'))!;
    const source = getGPUCommandNodeKernelSources([node])[0].source;
    expect(source).not.toContain('for (var slot');
    const parameters = createView.mock.results.find(
      result =>
        result.type === 'return' && result.value.buffer.id === 'study-window-sample-parameters'
    )!.value;
    expect(parameters.length).toBe(4);
    const memory = createCPUGraphMemory();
    memory.write(
      preparation.getViews(graph).summary,
      Float32Array.of(10, 20, 18, 26, 48, 28, 4, 0)
    );
    executeGeometryNodesOnCPU([node], memory);
    // Cell zero origin is (10,20); size is ((18-10)/4, (26-20)/2).
    expect(Array.from(memory.readFloat32(parameters))).toEqual([10, 20, 2, 3]);
  } finally {
    createView.mockRestore();
    preparation.destroy();
    device.destroy();
  }
});
