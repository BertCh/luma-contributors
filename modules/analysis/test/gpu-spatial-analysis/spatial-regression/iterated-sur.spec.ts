// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSeeminglyUnrelatedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression-operation';
import {CPUSeeminglyUnrelatedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './iterated-sur-fixtures.json';
import {expectIteratedSURResult, SUR_OVERFLOW_EXPECTED} from './iterated-sur-test-utils';

it('refreshes seeded zero-response residuals before reporting convergence', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const errors = new Float32Array(16);
  errors[0] = 1;
  errors[9] = 1;
  for (const Operation of [
    GPUSeeminglyUnrelatedRegressionOperation,
    CPUSeeminglyUnrelatedRegressionOperation
  ]) {
    const result = await runOperationGraph(device, {
      inputs: {
        predictors: {
          format: 'float32',
          values: Float32Array.from({length: 16}, (_, index) => (index % 2 ? 1 : -1))
        },
        responses: {format: 'float32', values: new Float32Array(16)},
        initial: {format: 'float32', values: errors}
      },
      operations: [
        new Operation({
          equationCount: 2,
          predictorCount: 1,
          rowCount: 8,
          predictors: 'predictors',
          responses: 'responses',
          initialResiduals: 'initial',
          iteration: {maximumIterations: 2, tolerance: 2 ** -126},
          output: {
            coefficients: 'coefficients',
            covariance: 'covariance',
            residualCovariance: 'residualCovariance',
            residuals: 'residuals',
            status: 'status',
            iteration: {status: {iterationCount: 'iterationCount'}}
          }
        })
      ],
      outputs: [
        'coefficients',
        'covariance',
        'residualCovariance',
        'residuals',
        'status',
        'iterationCount'
      ]
    });
    expect(Array.from(result['status'])).toEqual([3]);
    expect(Array.from(result['iterationCount'])).toEqual([1]);
    for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'])
      expect(result[name].every(Number.isNaN), name).toBe(true);
  }
});

for (const iterated of [false, true]) {
  it(`matches every finite large-covariance closed-form output on GPU: ${iterated ? 'iterated' : 'default'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const errors = new Float32Array(16);
    errors[0] = 2 ** 64;
    errors[9] = 2 ** 64;
    for (const Operation of [
      GPUSeeminglyUnrelatedRegressionOperation,
      CPUSeeminglyUnrelatedRegressionOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {
            format: 'float32',
            values: Float32Array.from({length: 16}, (_, index) => (index % 2 ? 1 : -1))
          },
          responses: {
            format: 'float32',
            values: Float32Array.from({length: 16}, (_, index) => (index % 2 ? 3 : 1))
          },
          initial: {format: 'float32', values: errors}
        },
        operations: [
          new Operation({
            equationCount: 2,
            predictorCount: 1,
            rowCount: 8,
            predictors: 'predictors',
            responses: 'responses',
            initialResiduals: 'initial',
            iteration: iterated ? {maximumIterations: 1} : undefined,
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              residualCovariance: 'residualCovariance',
              residuals: 'residuals',
              status: 'status',
              iteration: {status: {iterationCount: 'iterationCount'}}
            }
          })
        ],
        outputs: [
          'coefficients',
          'covariance',
          'residualCovariance',
          'residuals',
          'status',
          'iterationCount'
        ]
      });
      // The normalized system is perfectly conditioned; allow float32 solve rounding.
      expectIteratedSURResult(result, {...SUR_OVERFLOW_EXPECTED, status: [iterated ? 4 : 0]}, 5e-5);
    }
  });
}

for (const fixture of fixtures) {
  for (const maximumIterations of [undefined, 1, 2, 100]) {
    it(`matches independent NumPy ${fixture.name} SUR on GPU and CPU operations: ${maximumIterations ?? 'one-step default'}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const round = fixture.rounds.find(
        value => value.maximumIterations === (maximumIterations ?? 1)
      )!;
      for (const Operation of [
        GPUSeeminglyUnrelatedRegressionOperation,
        CPUSeeminglyUnrelatedRegressionOperation
      ]) {
        const result = await runOperationGraph(device, {
          inputs: {
            predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
            responses: {format: 'float32', values: Float32Array.from(fixture.responses)},
            ...(fixture.weights
              ? {
                  offsets: {
                    format: 'uint32' as const,
                    values: Uint32Array.from(fixture.weights.offsets)
                  },
                  neighbors: {
                    format: 'uint32' as const,
                    values: Uint32Array.from(fixture.weights.neighbors)
                  },
                  weights: {
                    format: 'float32' as const,
                    values: Float32Array.from(fixture.weights.weights)
                  }
                }
              : {mask: {format: 'uint32' as const, values: Uint32Array.from(fixture.mask)}})
          },
          operations: [
            new Operation({
              equationCount: fixture.equationCount,
              predictorCount: 2,
              rowCount: 40,
              predictors: 'predictors',
              responses: 'responses',
              mask: fixture.weights ? undefined : 'mask',
              iteration: maximumIterations ? {maximumIterations} : undefined,
              spatial: fixture.weights
                ? {
                    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
                    output: {parameters: 'parameters', status: 'spatialStatus'}
                  }
                : undefined,
              output: {
                coefficients: 'coefficients',
                covariance: 'covariance',
                residualCovariance: 'residualCovariance',
                residuals: 'residuals',
                status: 'status',
                iteration: {status: {iterationCount: 'iterationCount'}}
              }
            })
          ],
          outputs: [
            'coefficients',
            'covariance',
            'residualCovariance',
            'residuals',
            'status',
            'iterationCount',
            ...(fixture.weights ? ['parameters', 'spatialStatus'] : [])
          ]
        });
        // Normal equations in float32, with a strongly correlated covariance: 5e-3
        // relative to coefficients/residual magnitudes and covariance diagonal scale.
        expectIteratedSURResult(
          result,
          {...round.expected, status: maximumIterations ? round.expected.status : [0]},
          5e-3
        );
        if (fixture.parameters) {
          expect(Array.from(result['spatialStatus'])).toEqual([0, 0]);
          fixture.parameters.forEach((value, index) => {
            expect(Number.isFinite(result['parameters'][index])).toBe(true);
            expect(Math.abs(result['parameters'][index] - value)).toBeLessThan(5e-4);
          });
        }
      }
    });
  }
}

for (const failure of ['singular-covariance', 'excluded']) {
  it(`iterated SUR publishes ${failure} with zero completed rounds and NaN numeric outputs`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = fixtures[0];
    for (const Operation of [
      GPUSeeminglyUnrelatedRegressionOperation,
      CPUSeeminglyUnrelatedRegressionOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
          responses: {format: 'float32', values: Float32Array.from(fixture.responses)},
          initial: {format: 'float32', values: new Float32Array(120).fill(2)},
          mask: {
            format: 'uint32',
            values: failure === 'excluded' ? new Uint32Array(40) : new Uint32Array(40).fill(1)
          }
        },
        operations: [
          new Operation({
            equationCount: 3,
            predictorCount: 2,
            rowCount: 40,
            predictors: 'predictors',
            responses: 'responses',
            mask: 'mask',
            initialResiduals: 'initial',
            iteration: {maximumIterations: 2},
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              residualCovariance: 'residualCovariance',
              residuals: 'residuals',
              status: 'status',
              iteration: {status: {iterationCount: 'iterationCount'}}
            }
          })
        ],
        outputs: [
          'coefficients',
          'covariance',
          'residualCovariance',
          'residuals',
          'status',
          'iterationCount'
        ]
      });
      expect(Array.from(result['status'])).toEqual([failure === 'excluded' ? 1 : 3]);
      expect(Array.from(result['iterationCount'])).toEqual([0]);
      for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'])
        expect(result[name].every(Number.isNaN), name).toBe(true);
    }
  });
}

for (const maximumIterations of [undefined, 2]) {
  it(`spatial SUR suppresses all-island equations with invalid GM parameters: ${maximumIterations ?? 'default'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [
      GPUSeeminglyUnrelatedRegressionOperation,
      CPUSeeminglyUnrelatedRegressionOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: Float32Array.of(-1, 0, 1, 0, 1, 0)},
          responses: {format: 'float32', values: Float32Array.of(0, 1, 0, 0, 0, 1)},
          offsets: {format: 'uint32', values: new Uint32Array(4)},
          neighbors: {format: 'uint32', values: Uint32Array.of(0)},
          weights: {format: 'float32', values: Float32Array.of(0)}
        },
        operations: [
          new Operation({
            equationCount: 2,
            predictorCount: 1,
            rowCount: 3,
            predictors: 'predictors',
            responses: 'responses',
            iteration: maximumIterations ? {maximumIterations} : undefined,
            spatial: {
              weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
              output: {parameters: 'parameters', status: 'spatialStatus'}
            },
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              residualCovariance: 'residualCovariance',
              residuals: 'residuals',
              status: 'status',
              iteration: {status: {iterationCount: 'iterationCount'}}
            }
          })
        ],
        outputs: [
          'coefficients',
          'covariance',
          'residualCovariance',
          'residuals',
          'status',
          'iterationCount',
          'parameters',
          'spatialStatus'
        ]
      });
      // W=0 violates the row-standardized/no-island contract. Invalid lambda cannot
      // turn the zero-neighbor filter into a successful unfiltered fit.
      expect(Array.from(result['spatialStatus'])).toEqual([4, 4]);
      expect(Array.from(result['status'])).toEqual([1]);
      expect(Array.from(result['iterationCount'])).toEqual([0]);
      for (const name of [
        'coefficients',
        'covariance',
        'residualCovariance',
        'residuals',
        'parameters'
      ])
        expect(result[name].every(Number.isNaN), name).toBe(true);
    }
  });
}
