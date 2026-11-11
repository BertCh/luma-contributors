// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUOrdinaryLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation';
import {GPUOrdinaryLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares-operation';
import {getGPUOrdinaryLeastSquaresParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './ordinary-least-squares-operation-fixtures.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const scene of ['ordinary', 'ridge', 'excluded', 'singular'] as const) {
  it(`OLS operations agree on ${scene}, including every diagnostic and row output`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = fixtures[scene === 'ridge' ? 1 : 0];
    const outputNames = [
      'code',
      'coefficients',
      'standardErrors',
      'tStatistics',
      'summary',
      'residuals',
      'fitted'
    ];
    const outputs = [];
    for (const Operation of [GPUOrdinaryLeastSquaresOperation, CPUOrdinaryLeastSquaresOperation]) {
      outputs.push(
        await runOperationGraph(device, {
          inputs: {
            predictors: {
              format: 'float32',
              values:
                scene === 'singular'
                  ? new Float32Array(fixture.predictors.length)
                  : Float32Array.from(fixture.predictors)
            },
            response: {format: 'float32', values: Float32Array.from(fixture.response)},
            mask: {
              format: 'uint32',
              values: scene === 'excluded' ? new Uint32Array(32) : Uint32Array.from(fixture.mask)
            },
            parameters: {
              format: 'float32',
              values: getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: fixture.ridgeLambda})
            }
          },
          operations: [
            new Operation({
              rowCount: 32,
              predictorCount: 2,
              predictors: 'predictors',
              response: 'response',
              mask: 'mask',
              parameters: 'parameters',
              output: {
                status: {code: 'code'},
                coefficients: 'coefficients',
                standardErrors: 'standardErrors',
                tStatistics: 'tStatistics',
                summary: 'summary',
                residuals: 'residuals',
                fitted: 'fitted'
              }
            })
          ],
          outputs: outputNames
        })
      );
    }
    expect(Array.from(outputs[0]['code'])).toEqual(Array.from(outputs[1]['code']));
    for (const name of outputNames.slice(1)) {
      // Well-conditioned scenes: float32 residual moments and auxiliary fits dominate report error.
      expectRegressionRowsClose(
        outputs[0][name],
        outputs[1][name],
        name === 'summary' ? 5e-3 : 5e-4
      );
    }
    expect(outputs[1]['summary'][0]).toBe(
      scene === 'excluded' ? 0 : fixture.mask.filter(Boolean).length
    );
  });
}
