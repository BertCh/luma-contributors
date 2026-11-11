// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUProbitRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-probit-regression-operation';
import {GPUProbitRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import fixture from './probit-regression-operation-fixtures.json';

for (const scene of ['ordinary', 'excluded', 'budget', 'separated', 'zeroWeights'] as const) {
  it(`probit operations agree on every output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const outputNames = [
      'coefficients',
      'standardErrors',
      'fitted',
      'marginalEffects',
      'summary',
      'status',
      'tests',
      'diagnosticStatus'
    ];
    const results = [];
    for (const Operation of [GPUProbitRegressionOperation, CPUProbitRegressionOperation]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {
            predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
            response: {
              format: 'float32',
              values:
                scene === 'separated'
                  ? new Float32Array(96).fill(1)
                  : Float32Array.from(fixture.response)
            },
            mask: {
              format: 'uint32',
              values: scene === 'excluded' ? new Uint32Array(96) : Uint32Array.from(fixture.mask)
            },
            offsets: {format: 'uint32', values: Uint32Array.from(fixture.offsets)},
            neighbors: {format: 'uint32', values: Uint32Array.from(fixture.neighbors)},
            weights: {
              format: 'float32',
              values:
                scene === 'zeroWeights'
                  ? new Float32Array(fixture.weights.length)
                  : Float32Array.from(fixture.weights)
            }
          },
          operations: [
            new Operation({
              rowCount: 96,
              predictorCount: 2,
              predictors: 'predictors',
              response: 'response',
              mask: 'mask',
              maximumIterations: scene === 'budget' ? 1 : undefined,
              spatialDiagnostics: {
                weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
                output: {tests: 'tests', status: 'diagnosticStatus'}
              },
              output: {
                coefficients: 'coefficients',
                standardErrors: 'standardErrors',
                fitted: 'fitted',
                marginalEffects: 'marginalEffects',
                summary: 'summary',
                status: 'status'
              }
            })
          ],
          outputs: outputNames
        })
      );
    }
    for (const name of ['status', 'diagnosticStatus'])
      expect(Array.from(results[0][name])).toEqual(Array.from(results[1][name]));
    expect(results[1]['status'][0]).toBe(
      scene === 'excluded' ? 1 : scene === 'separated' ? 4 : scene === 'budget' ? 3 : 0
    );
    for (const name of ['coefficients', 'standardErrors', 'fitted', 'marginalEffects', 'tests']) {
      // Relative error only: float32 tile sums, erfc and Fisher steps dominate these scenes.
      expectRegressionRowsClose(results[0][name], results[1][name], 5e-4);
    }
    expectRegressionRowsClose(
      results[0]['summary'].subarray(0, 3),
      results[1]['summary'].subarray(0, 3),
      5e-4
    );
    // A float32 step can cross the stopping tolerance a round earlier/later. Both iteration
    // counts must stay within one round; the explicit one-round budget and failures are exact.
    const iterationDifference = Math.abs(results[0]['summary'][3] - results[1]['summary'][3]);
    expect(iterationDifference).toBeLessThanOrEqual(
      scene === 'ordinary' || scene === 'zeroWeights' ? 1 : 0
    );
  });
}
