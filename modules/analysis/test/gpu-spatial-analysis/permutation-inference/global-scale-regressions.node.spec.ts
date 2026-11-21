// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGlobalPermutationTest} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

it('UST-01 accumulates General G ordered off-diagonal mass on every global route', () => {
  for (const route of ['reference', 'tiled', 'permutation'] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const props = {
      weights: {
        offsets: view('offsets', 'uint32', 5),
        neighbors: view('neighbors', 'uint32', 6),
        weights: view('weights', 'float32', 6)
      },
      values: view('values', 'float32', 4),
      output: {summary: view('summary', 'float32', 64)}
    };
    const contributor =
      route === 'permutation'
        ? new GPUGlobalPermutationTest({
            ...props,
            parameters: view('parameters', 'uint32', 4),
            maximumPermutations: 9,
            statistic: 'getisOrdG'
          })
        : new GPUGlobalSpatialStatistics({...props, mode: route, statistics: ['getisOrdG']});
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    for (const {id, source} of sources)
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    expect(
      sources.some(({source}) => source.includes('leftSum * rightSum')),
      route
    ).toBe(true);
    if (route !== 'permutation')
      expect(sources.some(({id}) => id.endsWith('-g-product-moments-store'))).toBe(true);
    device.destroy();
  }
});

it('B1 normalizes bivariate columns with exact powers of two before centering', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const contributor = new GPUGlobalPermutationTest({
    weights: {
      offsets: view('offsets', 'uint32', 5),
      neighbors: view('neighbors', 'uint32', 6),
      weights: view('weights', 'float32', 6)
    },
    values: view('x', 'float32', 4),
    secondValues: view('y', 'float32', 4),
    parameters: view('parameters', 'uint32', 4),
    maximumPermutations: 9,
    statistic: 'bivariateMoran',
    output: {summary: view('summary', 'float32', 12)}
  });
  const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(node =>
    node.id.endsWith('-moment-inputs')
  )!.source;
  expect(source).toContain('getPermutationInverseScale');
  expect(source).not.toContain('/ max(valueScales');
  device.destroy();
});
