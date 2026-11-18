// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUOrdinaryLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation';
import {computeOrdinaryLeastSquaresOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation';
import {
  GPUOrdinaryLeastSquaresOperation,
  type OrdinaryLeastSquaresOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares-operation';
import {getGPUOrdinaryLeastSquaresParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './ordinary-least-squares-operation-fixtures.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const mutation of ['rename', 'remove', 'replace', 'add'] as const) {
  it(`OLS operations retain declared prediction output after caller ${mutation}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [GPUOrdinaryLeastSquaresOperation, CPUOrdinaryLeastSquaresOperation]) {
      const props: OrdinaryLeastSquaresOperationProps = {
        rowCount: 4,
        predictorCount: 1,
        predictors: 'predictors',
        response: 'response',
        output: {
          status: {code: 'code'},
          coefficients: 'coefficients',
          standardErrors: 'standardErrors',
          tStatistics: 'tStatistics',
          summary: 'summary',
          fitted: 'fitted',
          residuals: 'residuals',
          ...(mutation === 'add' ? {} : {predictions: 'heldOut'})
        }
      };
      const operation = new Operation(props);
      const declared = operation.contract.outputs;
      if (!declared || typeof declared === 'function')
        throw new Error('OLS outputs must be static');
      const outputNames = Object.keys(declared);
      if (mutation === 'rename') props.output.predictions = 'otherScene';
      else if (mutation === 'remove') delete props.output.predictions;
      else if (mutation === 'replace') props.output = {...props.output, predictions: 'otherScene'};
      else props.output.predictions = 'otherScene';
      props.output.status.code = 'otherStatus';
      props.output.coefficients = 'otherCoefficients';
      delete props.output.fitted;
      delete props.output.residuals;
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: Float32Array.of(0, 1, 2, 3)},
          response: {format: 'float32', values: Float32Array.of(1, 3, 5, 7)}
        },
        operations: [operation],
        outputs: outputNames
      });
      expect(Object.keys(result).sort()).toEqual(outputNames.sort());
      expect(Array.from(result['code'])).toEqual([0]);
      for (const [name, expected] of [
        ['coefficients', [1, 2]],
        ['fitted', [1, 3, 5, 7]],
        ['residuals', [0, 0, 0, 0]],
        ...(mutation === 'add' ? [] : [['heldOut', [1, 3, 5, 7]]])
      ] as [string, number[]][]) {
        expect(result[name].length).toBe(expected.length);
        expected.forEach((value, row) => {
          expect(Number.isFinite(result[name][row])).toBe(true);
          expect(result[name][row]).toBeCloseTo(value, 4);
        });
      }
      expect(result).not.toHaveProperty('otherScene');
    }
  });
}

for (const origin of [0, 5000000]) {
  it(`OLS origin-shifted tile means preserve a rare varying predictor at ${origin}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const trainingCount = 2048;
    const predictors = new Float32Array(trainingCount + 1).fill(origin);
    const response = new Float32Array(trainingCount + 1);
    const mask = new Uint32Array(trainingCount + 1).fill(1);
    predictors[0] = origin + 0.5;
    predictors[trainingCount] = origin + 0.5;
    response[0] = 1;
    response[trainingCount] = NaN;
    mask[trainingCount] = 0;
    const reference = computeOrdinaryLeastSquaresOnCPU({predictors, response, mask}, 1);
    expect(reference.status[0]).toBe(0);
    expect(reference.coefficients[1]).toBeCloseTo(2, 5);
    expect(reference.predictions[trainingCount]).toBeCloseTo(1, 5);
    for (const Operation of [GPUOrdinaryLeastSquaresOperation, CPUOrdinaryLeastSquaresOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: predictors},
          response: {format: 'float32', values: response},
          mask: {format: 'uint32', values: mask}
        },
        operations: [
          new Operation({
            rowCount: predictors.length,
            predictorCount: 1,
            tileRowCount: trainingCount,
            predictors: 'predictors',
            response: 'response',
            mask: 'mask',
            output: {
              status: {code: 'code'},
              coefficients: 'coefficients',
              standardErrors: 'standardErrors',
              tStatistics: 'tStatistics',
              summary: 'summary',
              predictions: 'predictions'
            }
          })
        ],
        outputs: ['code', 'coefficients', 'predictions']
      });
      expect(result['code'][0]).toBe(0);
      expect(result['coefficients'][1]).toBeCloseTo(2, 4);
      expect(result['predictions'][trainingCount]).toBeCloseTo(1, 4);
    }
  });
}

it('OLS operations preserve a representable raw intercept with rounded large means', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUOrdinaryLeastSquaresOperation, CPUOrdinaryLeastSquaresOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        predictors: {
          format: 'float32',
          values: Float32Array.of(5000000, 5000000.5, 5000000, 5000000.5)
        },
        response: {
          format: 'float32',
          values: Float32Array.of(5000000.5, 5000001, 5000000.5, 5000001)
        }
      },
      operations: [
        new Operation({
          rowCount: 4,
          predictorCount: 1,
          predictors: 'predictors',
          response: 'response',
          output: {
            status: {code: 'code'},
            coefficients: 'coefficients',
            standardErrors: 'standardErrors',
            tStatistics: 'tStatistics',
            summary: 'summary',
            fitted: 'fitted',
            predictions: 'predictions'
          }
        })
      ],
      outputs: ['code', 'coefficients', 'fitted', 'predictions']
    });
    expect(result['code'][0]).toBe(0);
    expect(result['coefficients'][0]).toBeCloseTo(0.5, 5);
    expect(result['coefficients'][1]).toBeCloseTo(1, 5);
    expect(Array.from(result['fitted'])).toEqual([5000000.5, 5000001, 5000000.5, 5000001]);
    expect(Array.from(result['predictions'])).toEqual([5000000.5, 5000001, 5000000.5, 5000001]);
  }
});

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
      'fitted',
      'predictions'
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
                fitted: 'fitted',
                predictions: 'predictions'
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

it('OLS operations predict held-out rows in centered coordinates while fitted rows honor the mask', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUOrdinaryLeastSquaresOperation, CPUOrdinaryLeastSquaresOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        predictors: {
          format: 'float32',
          values: Float32Array.of(4999998, 4999998, 5000002, 5000002, 5000000, NaN)
        },
        response: {format: 'float32', values: Float32Array.of(-15, -15, 17, 17, NaN, 1)},
        mask: {format: 'uint32', values: Uint32Array.of(1, 1, 1, 1, 0, 0)}
      },
      operations: [
        new Operation({
          rowCount: 6,
          predictorCount: 1,
          predictors: 'predictors',
          response: 'response',
          mask: 'mask',
          output: {
            status: {code: 'code'},
            coefficients: 'coefficients',
            standardErrors: 'standardErrors',
            tStatistics: 'tStatistics',
            summary: 'summary',
            fitted: 'fitted',
            predictions: 'predictions'
          }
        })
      ],
      outputs: ['code', 'fitted', 'predictions']
    });
    expect(result['code'][0]).toBe(0);
    [-15, -15, 17, 17, 1].forEach((expected, row) =>
      expect(result['predictions'][row]).toBeCloseTo(expected, 5)
    );
    expect(Number.isNaN(result['predictions'][5])).toBe(true);
    expect(Number.isNaN(result['fitted'][4])).toBe(true);
    expect(Number.isNaN(result['fitted'][5])).toBe(true);
  }
});
