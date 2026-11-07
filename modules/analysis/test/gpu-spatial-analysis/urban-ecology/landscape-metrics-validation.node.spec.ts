// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addLandscapeMetricsRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('landscape metrics validates inputs before reconcilePatches mutates the graph', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let addCount = 0;
    const add = graph.add.bind(graph);
    graph.add = ((...args: Parameters<typeof graph.add>) => {
      addCount++;
      return add(...args);
    }) as typeof graph.add;
    const patches = createTransientView(graph, 'patches', 'uint32', 8);
    const patchClasses = createTransientView(graph, 'patch-classes', 'uint32', 3);
    expect(() =>
      addLandscapeMetricsRecipe(graph, {
        patches,
        patchClasses,
        width: 4,
        height: 2,
        classCount: 0,
        cellSize: [1, 1],
        connectivity: 4,
        reconcilePatches: true
      })
    ).toThrow();
    expect(addCount).toBe(0);
  } finally {
    device.destroy();
  }
});
