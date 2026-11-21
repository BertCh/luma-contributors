// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUMultivariateLocalGearyPermutationTest,
  getGPUMultivariateLocalGearyPermutationTestParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  computeMultivariateGearyRuntimeOracle,
  MULTIVARIATE_GEARY_RUNTIME_SEED
} from './multivariate-geary-runtime-oracle';

for (const strategy of ['exact', 'tiled'] as const) {
  it(`${strategy} multivariate inference reads candidate-only loss before standardization`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `inference-audit-${serial++}`, format, length);
    try {
      const nodes = new GPUMultivariateLocalGearyPermutationTest({
        strategy,
        variableCount: 2,
        permutationCount: 9,
        values: view('float32', 16),
        weights: {
          offsets: view('uint32', 9),
          neighbors: view('uint32', 8),
          weights: view('float32', 8),
          status: {candidateOverflow: view('uint32', 1)}
        },
        output: {
          statistic: view('float32', 8),
          pseudoPValues: view('float32', 8),
          classes: view('uint32', 8)
        }
      }).getCommandNodes(graph);
      const sources = getGPUCommandNodeKernelSources(nodes);
      expect(
        sources.some(record => record.source.includes('lossInput0[lossInput0Offset]')),
        'candidate flag has a shader reader'
      ).toBe(true);
      expect(sources.find(record => record.id.endsWith('-complete'))!.source).toContain(
        'overflow[overflowOffset] == 0u'
      );
    } finally {
      device.destroy();
    }
  });
}

it('multivariate inference rejects significance values that round outside float32 (0,1)', () => {
  for (const significance of [1 - 2 ** -26, Number.MIN_VALUE])
    expect(() =>
      getGPUMultivariateLocalGearyPermutationTestParameterValues({significanceLevel: significance})
    ).toThrow(/significance/);
});

it('multivariate CPU twin agrees with independent arbitrary-integer Philox and rational distance expectations', () => {
  // multivariate-inference-audit.oracle.py uses arbitrary integers, not the production limb helper.
  for (const [mode, probabilities, classes] of [
    ['none', [0.5, 0.6, 0.9, 0.7, 0.6], [1, 1, 1, 1, 2]],
    ['bonferroni', [0.3, 1, 1, 1, 1], [1, 0, 0, 0, 0]],
    ['benjamini-hochberg', [0.875, 0.875, 0.9, 0.875, 0.875], [1, 1, 1, 1, 2]]
  ] as const) {
    const actual = computeMultivariateGearyRuntimeOracle(
      MULTIVARIATE_GEARY_RUNTIME_SEED,
      0.99,
      mode
    );
    for (const [row, expected] of [...probabilities, NaN, NaN, NaN].entries()) {
      if (Number.isNaN(expected)) expect(actual.pValues[row]).toBeNaN();
      else {
        expect(Number.isFinite(actual.pValues[row])).toBe(true);
        expect(Math.abs(actual.pValues[row] - expected)).toBeLessThan(1e-15);
      }
    }
    expect(actual.classes).toEqual([...classes, 0xffffffff, 0xffffffff, 0xffffffff]);
    expect(actual.statistic).toEqual(
      [1, 4, 16, 64, 225].map(value => value / 117.25).concat([NaN, NaN, NaN])
    );
    const squaredDraws = [
      961, 9, 961, 49, 1, 961, 9, 1, 49, 961, 225, 49, 9, 961, 961, 225, 1, 961, 1, 900, 1, 196, 36,
      196, 36, 900, 4, 4, 900, 196, 196, 36, 196, 36, 1, 36, 1, 196, 144, 16, 144, 144, 144, 784, 4,
      9, 16, 16, 784, 9, 144, 144, 16, 784, 784, 4, 784, 36, 576, 49, 64, 36, 64, 49, 36, 36, 64,
      64, 36, 36, 16, 16, 16, 49, 16, 64, 196, 144, 64, 256, 225, 256, 144, 144, 196, 144, 64, 144,
      256, 64, 256, 196, 196, 196, 196
    ];
    expect(actual.simulations).toEqual(
      squaredDraws.map(value => value / 117.25).concat(new Array(57).fill(NaN))
    );
  }
});
