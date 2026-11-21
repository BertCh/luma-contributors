// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkAccessibility,
  getGPUNetworkAccessibilityParameterValues
} from '../../../src/gpu-network/network-accessibility/gpu-network-accessibility';
import {
  GPUNetworkCostWeights,
  getGPUNetworkCostWeightsParameterValues
} from '../../../src/gpu-network/network-cost-weights/gpu-network-cost-weights';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

for (const decay of ['exponential', 'power'] as const) {
  it(`DEF-08 ${decay} costs 1 and 9 preserve explicit beta zero and one on replay`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const context = createOracleContext(device);
    const {graph, input, output, write, read} = context;
    const costs = input('costs', 'float32', [1, 9]);
    const weightsParameters = input(
      'weights-parameters',
      'float32',
      Array.from(getGPUNetworkCostWeightsParameterValues({costLimit: 10, decay, beta: 0}))
    );
    const scoringParameters = input(
      'scoring-parameters',
      'float32',
      Array.from(getGPUNetworkAccessibilityParameterValues({threshold: 10, decay, beta: 0}))
    );
    const weights = output('weights', 'float32', 2);
    const gravity = output('gravity', 'float32');
    try {
      graph.add(
        new GPUNetworkCostWeights({
          costs,
          rowCount: 1,
          parameters: weightsParameters,
          output: {
            offsets: output('offsets', 'uint32', 2),
            neighbors: output('neighbors', 'uint32', 2),
            weights
          }
        })
      );
      graph.add(
        new GPUNetworkAccessibility({
          costs,
          orientation: 'origin-rows',
          opportunityWeights: input('opportunities', 'float32', [1, 1]),
          parameters: scoringParameters,
          gravity
        })
      );
      for (const beta of [0, 1, 0]) {
        write(
          weightsParameters,
          Array.from(getGPUNetworkCostWeightsParameterValues({costLimit: 10, decay, beta}))
        );
        write(
          scoringParameters,
          Array.from(getGPUNetworkAccessibilityParameterValues({threshold: 10, decay, beta}))
        );
        context.run();
        // exp(-1), exp(-9), and 1^-1, 9^-1 from the formulas, independent of producer code.
        const expected =
          beta === 0
            ? [1, 1]
            : decay === 'exponential'
              ? [0.36787944117144233, 0.00012340980408667956]
              : [1, 0.1111111111111111];
        const actual = await read(weights);
        expect(actual).toHaveLength(expected.length);
        expect(actual.every(Number.isFinite)).toBe(true);
        actual.forEach((value, index) =>
          expect(Math.abs(value - expected[index])).toBeLessThan(2e-6)
        );
        const scores = await read(gravity);
        expect(scores).toHaveLength(1);
        const score = scores[0];
        expect(Number.isFinite(score)).toBe(true);
        expect(Math.abs(score - (expected[0] + expected[1]))).toBeLessThan(4e-6);
      }
    } finally {
      context.destroy();
    }
  });
}
