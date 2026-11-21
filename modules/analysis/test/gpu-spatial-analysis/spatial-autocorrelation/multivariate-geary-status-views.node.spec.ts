// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUMultivariateLocalGearyPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-multivariate-local-geary-permutation-test';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const invalidSource of ['aliased-output', 'foreign-graph'] as const) {
  it(`multivariate Geary rejects ${invalidSource} weights overflow before graph execution`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const foreignGraph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `geary-status-${serial++}`, format, length);
    const classes = view('uint32', 4);
    const overflow =
      invalidSource === 'aliased-output'
        ? graph.createDataView(classes.buffer, {format: 'uint32', length: 1})
        : createTransientView(foreignGraph, 'foreign-overflow', 'uint32', 1);
    try {
      expect(() =>
        new GPUMultivariateLocalGearyPermutationTest({
          values: view('float32', 8),
          variableCount: 2,
          permutationCount: 3,
          weights: {
            offsets: view('uint32', 5),
            neighbors: view('uint32', 8),
            weights: view('float32', 8),
            status: {overflow}
          },
          output: {statistic: view('float32', 4), pseudoPValues: view('float32', 4), classes}
        }).getCommandNodes(graph)
      ).toThrow(
        invalidSource === 'aliased-output'
          ? /must not share buffers with inputs/
          : /must belong to the target graph/
      );
    } finally {
      device.destroy();
    }
  });
}
