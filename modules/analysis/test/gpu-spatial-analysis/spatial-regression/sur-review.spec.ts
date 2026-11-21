// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSeeminglyUnrelatedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression-operation';
import {CPUSeeminglyUnrelatedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectIteratedSURResult} from './iterated-sur-test-utils';
import seeded from './sur-review-fixtures.json';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getGPUSeeminglyUnrelatedRegressionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/seemingly-unrelated-regression-parameters';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

it('R2 retains binary residual noise of 2^-16 relative to a unit fitted signal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const amplitude = 2 ** -16;
  const predictors = Float32Array.from(
    {length: 16},
    (_, index) => 2 * ((index < 8 ? index : Math.floor(index / 2)) % 2) - 1
  );
  const responses = Float32Array.from({length: 16}, (_, index) => {
    const row = index % 8,
      first = 2 * (row % 2) - 1,
      second = 2 * (Math.floor(row / 2) % 2) - 1,
      third = 2 * (Math.floor(row / 4) % 2) - 1;
    return index < 8
      ? 1 + first + amplitude * (second + first * second)
      : 2 + second + amplitude * (0.5 * first + first * second + third);
  });
  for (const Operation of [
    GPUSeeminglyUnrelatedRegressionOperation,
    CPUSeeminglyUnrelatedRegressionOperation
  ]) {
    const result = await runOperationGraph(device, {
      inputs: {
        predictors: {format: 'float32', values: predictors},
        responses: {format: 'float32', values: responses}
      },
      operations: [
        new Operation({
          equationCount: 2,
          predictorCount: 1,
          rowCount: 8,
          predictors: 'predictors',
          responses: 'responses',
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
    expect(Array.from(result['status'])).toEqual([0]);
    expect(Array.from(result['iterationCount'])).toEqual([1]);
    [2, 1, 1, 2.25].forEach((value, index) => {
      expect(Number.isFinite(result['residualCovariance'][index])).toBe(true);
      expect(
        Math.abs(result['residualCovariance'][index] / (value * amplitude ** 2) - 1)
      ).toBeLessThan(0.01);
    });
  }
});

it('R9 changes tolerance on one compiled GPU graph without changing its command graph', async () => {
  const context = await createRegressionTestContext();
  const parameters = new GPUParameterBuffer(context.device, {
    id: 'sur-tolerance',
    format: 'float32',
    length: 1,
    values: getGPUSeeminglyUnrelatedRegressionParameterValues({tolerance: 0.5})
  });
  const output = {
    coefficients: context.output('float32', 4),
    covariance: context.output('float32', 16),
    residualCovariance: context.output('float32', 4),
    residuals: context.output('float32', 16),
    status: context.output('uint32', 1),
    iterationCount: context.output('uint32', 1)
  };
  context.graph.add(
    new GPUSeeminglyUnrelatedRegression({
      equationCount: 2,
      predictorCount: 1,
      rowCount: 8,
      predictors: context.input('float32', seeded.predictors),
      responses: context.input('float32', seeded.responses),
      initialResiduals: context.input('float32', seeded.initialResiduals),
      parameters: parameters.importToGraph(context.graph),
      iteration: {maximumIterations: 100},
      output: {
        coefficients: output.coefficients.view,
        covariance: output.covariance.view,
        residualCovariance: output.residualCovariance.view,
        residuals: output.residuals.view,
        status: output.status.view,
        iteration: {status: {iterationCount: output.iterationCount.view}}
      }
    })
  );
  const compiled = context.graph.compile();
  try {
    for (const tolerance of [0.5, 1e-5, 0.5, 0]) {
      parameters.write(
        tolerance
          ? getGPUSeeminglyUnrelatedRegressionParameterValues({tolerance})
          : Float32Array.of(0)
      );
      submitGraph(context.device, compiled, undefined);
      const actual = {
        coefficients: await readFloat32(output.coefficients.buffer),
        covariance: await readFloat32(output.covariance.buffer),
        residualCovariance: await readFloat32(output.residualCovariance.buffer),
        residuals: await readFloat32(output.residuals.buffer),
        status: await readUint32(output.status.buffer),
        iterationCount: await readUint32(output.iterationCount.buffer)
      };
      if (tolerance === 0) {
        expect(actual.status).toEqual([5]);
        expect(actual.iterationCount).toEqual([0]);
        for (const name of [
          'coefficients',
          'covariance',
          'residualCovariance',
          'residuals'
        ] as const)
          expect(actual[name].every(Number.isNaN)).toBe(true);
      } else
        expectIteratedSURResult(
          actual,
          {...seeded.rounds[tolerance === 0.5 ? 1 : 2].expected, status: [0]},
          5e-4
        );
    }
  } finally {
    compiled.destroy();
    parameters.destroy();
    context.destroy();
  }
});

it('R8 every final spatial output differs visibly from the preceding round', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = seeded.spatialFinal;
  for (const Operation of [
    GPUSeeminglyUnrelatedRegressionOperation,
    CPUSeeminglyUnrelatedRegressionOperation
  ]) {
    const result = await runOperationGraph(device, {
      inputs: {
        predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
        responses: {format: 'float32', values: Float32Array.from(fixture.responses)},
        offsets: {format: 'uint32', values: Uint32Array.from(fixture.weights.offsets)},
        neighbors: {format: 'uint32', values: Uint32Array.from(fixture.weights.neighbors)},
        weights: {format: 'float32', values: Float32Array.from(fixture.weights.weights)}
      },
      operations: [
        new Operation({
          equationCount: 3,
          predictorCount: 2,
          rowCount: 40,
          predictors: 'predictors',
          responses: 'responses',
          spatial: {
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            output: {parameters: 'parameters', status: 'spatialStatus'}
          },
          iteration: {maximumIterations: 100, tolerance: fixture.tolerance},
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
    // Round 2 changes slopes by ~0.01 and residuals by ~0.018; 5e-4 allows
    // float32 moment/solve rounding but rejects the entire preceding numeric iterate.
    expectIteratedSURResult(result, fixture.expected, 5e-4);
    expect(Array.from(result['spatialStatus'])).toEqual([0, 0, 0]);
    fixture.expected.parameters.forEach((value, index) => {
      expect(Number.isFinite(result['parameters'][index])).toBe(true);
      expect(Math.abs(result['parameters'][index] - value)).toBeLessThan(5e-4);
    });
  }
});

for (const shift of [0, 65536])
  it(`R5 normalized spatial GLS matches the independent shifted oracle: ${shift}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = seeded.spatialShift;
    for (const Operation of [
      GPUSeeminglyUnrelatedRegressionOperation,
      CPUSeeminglyUnrelatedRegressionOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
          responses: {
            format: 'float32',
            values: Float32Array.from(fixture.responses, value => value + shift)
          },
          offsets: {format: 'uint32', values: Uint32Array.from(fixture.weights.offsets)},
          neighbors: {format: 'uint32', values: Uint32Array.from(fixture.weights.neighbors)},
          weights: {format: 'float32', values: Float32Array.from(fixture.weights.weights)}
        },
        operations: [
          new Operation({
            equationCount: 2,
            predictorCount: 2,
            rowCount: 40,
            predictors: 'predictors',
            responses: 'responses',
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
      const expected = {
        ...fixture.expected,
        coefficients: fixture.expected.coefficients.map(
          (value, index) => value + (index % 3 === 0 ? shift : 0)
        )
      };
      expectIteratedSURResult(result, expected, 5e-4);
      expect(Array.from(result['spatialStatus'])).toEqual([0, 0]);
      for (const index of [0, 3])
        expect(
          Math.abs(result['coefficients'][index] - Math.fround(expected.coefficients[index]))
        ).toBeLessThanOrEqual(0.008);
      for (const index of [1, 2, 4, 5])
        expect(Math.abs(result['coefficients'][index] - expected.coefficients[index])).toBeLessThan(
          5e-4
        );
      fixture.expected.parameters.forEach((value, index) =>
        expect(Math.abs(result['parameters'][index] - value)).toBeLessThan(5e-4)
      );
    }
  });

it('R2 retains small but resolved equation variance after response rescaling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [
    GPUSeeminglyUnrelatedRegressionOperation,
    CPUSeeminglyUnrelatedRegressionOperation
  ]) {
    const result = await runOperationGraph(device, {
      inputs: {
        predictors: {format: 'float32', values: Float32Array.from(seeded.predictors)},
        responses: {
          format: 'float32',
          values: Float32Array.from(seeded.responses, value => value * 1e-12)
        }
      },
      operations: [
        new Operation({
          equationCount: 2,
          predictorCount: 1,
          rowCount: 8,
          predictors: 'predictors',
          responses: 'responses',
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
    expect(Array.from(result['status'])).toEqual([0]);
    expect(Array.from(result['iterationCount'])).toEqual([1]);
    [2e-24, 1e-24, 1e-24, 2.25e-24].forEach((value, index) => {
      expect(Number.isFinite(result['residualCovariance'][index])).toBe(true);
      expect(Math.abs(result['residualCovariance'][index] / value - 1)).toBeLessThan(2e-4);
    });
  }
});

for (const maximumIterations of [undefined, 1, 2]) {
  it(`R2 exact equation is singular before any GLS solve: ${maximumIterations ?? 'legacy'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [
      GPUSeeminglyUnrelatedRegressionOperation,
      CPUSeeminglyUnrelatedRegressionOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {
            format: 'float32',
            values: Float32Array.from({length: 16}, (_, index) =>
              index < 8 ? index : (3 * (index - 8)) % 7
            )
          },
          responses: {
            format: 'float32',
            values: Float32Array.from({length: 16}, (_, index) =>
              index < 8 ? 1 + 2 * index : (index - 7) % 3
            )
          }
        },
        operations: [
          new Operation({
            equationCount: 2,
            predictorCount: 1,
            rowCount: 8,
            predictors: 'predictors',
            responses: 'responses',
            iteration: maximumIterations ? {maximumIterations} : undefined,
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
      expect(Array.from(result['iterationCount'])).toEqual([0]);
      for (const name of ['coefficients', 'covariance', 'residuals'])
        expect(result[name].every(Number.isNaN)).toBe(true);
      if (maximumIterations) expect(result['residualCovariance'].every(Number.isNaN)).toBe(true);
      else {
        expect(Array.from(result['residualCovariance'].subarray(0, 3))).toEqual([0, 0, 0]);
        expect(Math.abs(result['residualCovariance'][3] - 689 / 1148)).toBeLessThan(1e-5);
      }
    }
  });
}

for (const round of seeded.rounds) {
  it(`R1 seeded SUR requires current residual covariance, budget ${round.maximumIterations}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [
      GPUSeeminglyUnrelatedRegressionOperation,
      CPUSeeminglyUnrelatedRegressionOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: Float32Array.from(seeded.predictors)},
          responses: {format: 'float32', values: Float32Array.from(seeded.responses)},
          initial: {format: 'float32', values: Float32Array.from(seeded.initialResiduals)}
        },
        operations: [
          new Operation({
            equationCount: 2,
            predictorCount: 1,
            rowCount: 8,
            predictors: 'predictors',
            responses: 'responses',
            initialResiduals: 'initial',
            iteration: {maximumIterations: round.maximumIterations},
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
      expectIteratedSURResult(result, round.expected, 5e-4);
    }
  });
}
