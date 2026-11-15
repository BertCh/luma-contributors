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

it.each([
  false,
  true
])('landscape metrics rejects aliased patch and class outputs before appending nodes (reconcilePatches=%s)', reconcilePatches => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const patches = createTransientView(graph, 'patches', 'uint32', 3);
    const patchClasses = createTransientView(graph, 'patch-classes', 'uint32', 2);
    const sharedMetrics = createTransientView(graph, 'shared-metrics', 'float32x4', 2);
    expect(() =>
      addLandscapeMetricsRecipe(graph, {
        patches,
        patchClasses,
        width: 3,
        height: 1,
        classCount: 2,
        cellSize: [1, 1],
        connectivity: 4,
        reconcilePatches,
        output: {patchMetrics: sharedMetrics, classMetrics: sharedMetrics}
      })
    ).toThrow(/^landscape-metrics outputs must not share a buffer$/);
    expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('landscape metrics rejects input/output buffer aliases before appending nodes (reconcilePatches=%s)', reconcilePatches => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const patches = createTransientView(graph, 'patches', 'uint32', 3);
    const patchClasses = createTransientView(graph, 'patch-classes', 'uint32', 2);
    const aggregation = graph.createDataView(patchClasses.buffer, {format: 'float32', length: 2});
    expect(() =>
      addLandscapeMetricsRecipe(graph, {
        patches,
        patchClasses,
        width: 3,
        height: 1,
        classCount: 2,
        cellSize: [1, 1],
        connectivity: 4,
        reconcilePatches,
        output: {aggregation}
      })
    ).toThrow(/^landscape-metrics outputs must not share buffers with inputs$/);
    expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});
