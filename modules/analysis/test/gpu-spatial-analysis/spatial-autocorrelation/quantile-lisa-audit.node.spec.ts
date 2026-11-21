// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {computeQuantileLISAReference} from '../recipes/quantile-lisa-reference';

it('quantile CPU twin matches type-five tie, mask and affine expectations on every port', () => {
  for (const scale of [1, 2]) {
    const values = Float32Array.from(
      [1, 1, 2, 3, 4, 5, 6, 7, 8, 999, NaN],
      value => scale * value + (scale === 2 ? 32 : 0)
    );
    const mask = Uint32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]);
    const result = computeQuantileLISAReference(values, 3, 2, mask);
    // NumPy hazen: h=n*p-.5 gives ranks 2.5 and 5.5 in [1,1,2,3,4,5,6,7,8].
    expect(result.boundaries).toEqual(scale === 1 ? [2.5, 5.5] : [37, 43]);
    expect(Array.from(result.membership)).toEqual([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]);
    expect(Array.from(result.validMask)).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
    expect(result.validCount).toBe(9);
  }
});

it('quantile LISA forwards its inherited exact analysisCount port to inference', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `quantile-audit-${serial++}`, format, length);
  try {
    const contributor = new GPUQuantileLISA({
      alternative: 'geoda-legacy',
      values: view('float32', 11),
      mask: view('uint32', 11),
      weights: {
        offsets: view('uint32', 12),
        neighbors: view('uint32', 16),
        weights: view('float32', 16)
      },
      quantileParameters: view('uint32', 2),
      parameters: view('uint32', 4),
      maximumPermutations: 9,
      output: {
        membership: view('uint32', 11),
        boundaries: view('float32', 2),
        validCount: view('uint32', 1),
        classes: view('uint32', 11),
        invalidQuantile: view('uint32', 1),
        exceedances: view('uint32', 11),
        pseudoPValues: view('float32', 11),
        analysisCount: view('uint32', 1),
        status: {overflow: view('uint32', 1)}
      }
    });
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(
      sources.some(record => record.source.includes('analysisCount[analysisCountOffset]')),
      'caller count port has a GPU writer'
    ).toBe(true);
  } finally {
    device.destroy();
  }
});
