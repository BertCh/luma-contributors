// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  CPUSpatialErrorMaximumLikelihoodOperation,
  type CPUSpatialErrorMaximumLikelihoodInput,
  type CPUSpatialErrorMaximumLikelihoodOptions
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-maximum-likelihood-operation';
import {GPUSpatialErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood-operation';
import {getGPUSpatialWeightsTracesParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import fixtures from './spatial-error-maximum-likelihood-operation-fixtures.json';

function createReviewCompleteWeights(rowCount: number) {
  return {
    offsets: Uint32Array.from({length: rowCount + 1}, (_, row) => row * (rowCount - 1)),
    neighbors: Uint32Array.from({length: rowCount * (rowCount - 1)}, (_, slot) => {
      const row = Math.floor(slot / (rowCount - 1));
      const column = slot % (rowCount - 1);
      return column < row ? column : column + 1;
    }),
    weights: new Float32Array(rowCount * (rowCount - 1)).fill(1)
  };
}

async function runReviewOperations(
  input: CPUSpatialErrorMaximumLikelihoodInput,
  options: CPUSpatialErrorMaximumLikelihoodOptions
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const results = [];
  for (const Operation of [
    GPUSpatialErrorMaximumLikelihoodOperation,
    CPUSpatialErrorMaximumLikelihoodOperation
  ]) {
    results.push(
      await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: input.weights.offsets},
          neighbors: {format: 'uint32', values: input.weights.neighbors},
          weights: {format: 'float32', values: input.weights.weights},
          predictors: {format: 'float32', values: input.predictors},
          response: {format: 'float32', values: input.response},
          ...(input.eigenvalues
            ? {eigenvalues: {format: 'float32' as const, values: input.eigenvalues}}
            : {})
        },
        operations: [
          new Operation({
            ...options,
            rowCount: input.response.length,
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            predictors: 'predictors',
            response: 'response',
            eigenvalues: input.eigenvalues ? 'eigenvalues' : undefined,
            output: {
              status: {code: 'code', converged: 'converged'},
              table: 'table',
              summary: 'summary',
              residuals: 'residuals',
              powerTraceStandardErrors: 'errors'
            }
          })
        ],
        outputs: ['code', 'converged', 'table', 'summary', 'residuals', 'errors']
      })
    );
  }
  return results;
}

it('review 1: constant orthogonal prediction has undefined pseudo R squared', async () => {
  const results = await runReviewOperations(
    {
      weights: createReviewCompleteWeights(4),
      predictors: Float32Array.of(-3, -1, 1, 3),
      response: Float32Array.of(0, 1, 1, 0),
      eigenvalues: Float32Array.of(-1, -1, -1, 3)
    },
    {predictorCount: 1, maximumPower: 16, logDeterminant: 'exact'}
  );
  for (const result of results) {
    expect(result['code'][0]).toBe(0);
    expect(result['summary'][6]).toBeNaN();
    expect(result['table'][4] === 0).toBe(true);
    expectRegressionRowsClose(result['residuals'], [-0.5, 0.5, 0.5, -0.5], 0);
  }
  for (const port of ['code', 'converged', 'table', 'summary', 'residuals', 'errors'])
    expectRegressionRowsClose(
      results[0][port],
      results[1][port],
      port === 'code' || port === 'converged' ? 0 : 5e-3
    );
});

it('rejects the exact-linear response with no finite Gaussian maximum in both operation forms', async () => {
  const values = Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7);
  const [gpu, cpu] = await runReviewOperations(
    {
      weights: createReviewCompleteWeights(8),
      predictors: values,
      response: values
    },
    {predictorCount: 1, maximumPower: 8}
  );
  // There is no finite Gaussian ML fit: b=(0,1) makes the variance exactly zero.
  // A rounding-created positive variance must not certify this unbounded likelihood.
  for (const result of [gpu, cpu]) {
    expect(result['code'][0]).toBe(3);
    expect(result['converged'][0]).toBe(0);
    expect(result['summary'][0]).toBe(8);
    expect(result['summary'].subarray(1).every(Number.isNaN)).toBe(true);
    expect(result['table'].every(Number.isNaN)).toBe(true);
    expect(result['residuals'].every(Number.isNaN)).toBe(true);
    expect(Array.from(result['errors'])).toEqual(new Array(8).fill(0));
  }
});

it('the scalar-identity flat profile uses the lower-bound tie and undefined spatial inference', async () => {
  const [gpu, cpu] = await runReviewOperations(
    {
      weights: {
        offsets: Uint32Array.of(0, 1, 2, 3, 4),
        neighbors: Uint32Array.of(0, 1, 2, 3),
        weights: Float32Array.of(3, 3, 3, 3)
      },
      predictors: Float32Array.of(0, 1, 2, 3),
      response: Float32Array.of(0, 1, 2, 4),
      eigenvalues: Float32Array.of(3, 3, 3, 3)
    },
    {predictorCount: 1, maximumPower: 16, logDeterminant: 'exact'}
  );
  expect(gpu['code'][0]).toBe(0);
  expect(cpu['code'][0]).toBe(0);
  expect(cpu['summary'][1]).toBe(Math.fround(-0.33));
  // Filtering scales every design/response row by 1-3lambda. Coefficients, their SEs,
  // residuals, conditioning and LL are invariant. The structural certificate fixes the
  // lower-parameter tie and suppresses inference on the unidentifiable lambda.
  for (const result of [gpu, cpu]) {
    expect(result['summary'][1]).toBe(Math.fround(-0.33));
    expect(result['summary'][16]).toBe(1);
    expect(result['summary'][19]).toBe(0);
    expect(result['summary'][20]).toBe(1);
    expect(result['converged'][0]).toBe(0);
    expect(result['table'].subarray(9, 12).every(Number.isNaN)).toBe(true);
  }
  expectRegressionRowsClose(gpu['table'].subarray(0, 8), cpu['table'].subarray(0, 8), 5e-3);
  expectRegressionRowsClose(gpu['residuals'], cpu['residuals'], 5e-3);
  for (const slot of [0, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18])
    expectRegressionRowsClose([gpu['summary'][slot]], [cpu['summary'][slot]], 5e-3);
  expect(Math.abs(gpu['summary'][1])).toBeLessThanOrEqual(Math.fround(0.33));
  const scaling = 1 - 3 * gpu['summary'][1];
  expectRegressionRowsClose(
    [gpu['summary'][2], gpu['summary'][21]],
    [0.075 * scaling ** 2, 4 * Math.log(scaling)],
    5e-3
  );
  expectRegressionRowsClose(gpu['errors'], cpu['errors'], 0);
});

for (const scene of [
  'ordinary-series',
  'series-boundary',
  'tail',
  'stochastic-seed-42',
  'eigenvalues',
  'signed-directed-series',
  'singular',
  'zero-links',
  'too-few-rows'
]) {
  it(`spatial error likelihood operations compare every output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = fixtures.find(value => value.name === scene) ?? fixtures[0];
    const tooFew = scene === 'too-few-rows';
    const rowCount = tooFew ? 3 : fixture.response.length;
    const names = ['code', 'converged', 'table', 'summary', 'residuals', 'errors'];
    const results = [];
    // Each run owns/destroys its operation; always construct a fresh instance.
    for (const Operation of [
      GPUSpatialErrorMaximumLikelihoodOperation,
      CPUSpatialErrorMaximumLikelihoodOperation
    ]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {
            offsets: {
              format: 'uint32',
              values: tooFew
                ? Uint32Array.of(0, 1, 2, 3)
                : Uint32Array.from(fixture.weights.offsets)
            },
            neighbors: {
              format: 'uint32',
              values: tooFew ? Uint32Array.of(1, 2, 0) : Uint32Array.from(fixture.weights.neighbors)
            },
            weights: {
              format: 'float32',
              values: tooFew
                ? Float32Array.of(1, 1, 1)
                : scene === 'zero-links'
                  ? new Float32Array(fixture.weights.weights.length)
                  : Float32Array.from(fixture.weights.weights)
            },
            predictors: {
              format: 'float32',
              values: tooFew
                ? Float32Array.of(0, 1, 1, 0, 0, 2)
                : scene === 'singular'
                  ? new Float32Array(fixture.predictors.length)
                  : Float32Array.from(fixture.predictors)
            },
            response: {
              format: 'float32',
              values: tooFew ? Float32Array.of(1, 2, 4) : Float32Array.from(fixture.response)
            },
            traceParameters: {
              format: 'uint32',
              values: getGPUSpatialWeightsTracesParameterValues({seed: 42})
            },
            ...(fixture.eigenvalues
              ? {
                  eigenvalues: {
                    format: 'float32' as const,
                    values: Float32Array.from(fixture.eigenvalues)
                  }
                }
              : {})
          },
          operations: [
            new Operation({
              ...fixture.options,
              rowCount,
              predictorCount: 2,
              traceMode: fixture.options.traceMode as 'exact' | 'stochastic',
              logDeterminant: fixture.options.logDeterminant as 'exact' | 'series',
              weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
              predictors: 'predictors',
              response: 'response',
              traceParameters: 'traceParameters',
              eigenvalues: fixture.eigenvalues ? 'eigenvalues' : undefined,
              output: {
                status: {code: 'code', converged: 'converged'},
                table: 'table',
                summary: 'summary',
                residuals: 'residuals',
                powerTraceStandardErrors: 'errors'
              }
            })
          ],
          outputs: names
        })
      );
    }
    const expectedCode = scene === 'singular' ? 1 : scene === 'zero-links' ? 3 : tooFew ? 2 : 0;
    for (const result of results) expect(Array.from(result['code'])).toEqual([expectedCode]);
    // Relative tolerance 5e-3 for floating quantities, including small p-values and residuals;
    // no absolute floor permits a small positive quantity to pass as zero. Flags/codes are exact.
    for (const name of names)
      expectRegressionRowsClose(
        results[0][name],
        results[1][name],
        name === 'code' || name === 'converged' ? 0 : 5e-3
      );
    for (const slot of [0, 12, 14, 15, 16, 17, 19, 20]) {
      if (expectedCode === 0 || slot === 0)
        expect(results[0]['summary'][slot] === results[1]['summary'][slot]).toBe(true);
    }
    if (expectedCode === 0) {
      // Compare each backend directly with the independent dense NumPy/SciPy literals.
      for (const result of results) {
        for (const port of ['table', 'summary', 'residuals', 'powerTraceStandardErrors'] as const) {
          expectRegressionRowsClose(
            result[port === 'powerTraceStandardErrors' ? 'errors' : port],
            Float32Array.from(fixture.expected[port], value => value ?? NaN),
            5e-3
          );
        }
        expect(result['converged'][0]).toBe(fixture.expected.summary[15]);
      }
      expect(Math.abs(results[1]['summary'][1])).toBeGreaterThan(0.05);
      expect(results[1]['table'][15]).toBeGreaterThan(0.01);
      expect(results[1]['table'][15]).toBeLessThan(0.9);
    }
  });
}
