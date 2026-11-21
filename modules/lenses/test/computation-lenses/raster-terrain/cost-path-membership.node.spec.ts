// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {GPUCommandGraph, GPUHistogram, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../../../analysis/src/testing';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {runExample} from '../../../src/computation-lenses/scenarios/gpu-raster/cost-distance-path.example';

vi.mock(
  '@luma-contributors/analysis/raster-analysis',
  () => import('../../../../analysis/src/raster-analysis')
);
vi.mock(
  '@luma-contributors/analysis/analysis-core',
  () => import('../../../../analysis/src/analysis-core')
);

test('W5-16 histogram selects the count-bounded path prefix', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const add = vi.spyOn(graph, 'add');
  const integers = (name: string, length = 300) =>
    createTransientView(graph, name, 'uint32', length);
  const path = {
    ids: integers('ids'),
    status: {count: integers('count', 1), overflow: integers('overflow', 1)}
  };
  const {parameters} = await runExample(device, {
    graph,
    grid: makeGPURasterMetadata({width: 300, height: 1}),
    friction: createTransientView(graph, 'friction', 'float32', 300),
    sourceMask: integers('source'),
    costs: createTransientView(graph, 'costs', 'float32', 300),
    backLinks: integers('links'),
    converged: integers('converged', 1),
    iterationCount: integers('iterations', 1),
    targets: [{target: integers('target', 1), path, mask: integers('membership')}]
  });
  try {
    const histogram = add.mock.calls
      .map(([producer]) => producer)
      .find(producer => producer instanceof GPUHistogram) as GPUHistogram;
    expect(histogram.mask).toBeDefined();
    expect((histogram.mask as {length: number}).length).toBe(path.ids.length);
  } finally {
    parameters.destroy();
    device.destroy();
  }
});
