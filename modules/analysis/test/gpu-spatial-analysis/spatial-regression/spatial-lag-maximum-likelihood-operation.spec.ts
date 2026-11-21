// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpatialLagMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-maximum-likelihood-operation';
import {GPUSpatialLagMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood-operation';
import {getGPUSpatialWeightsTracesParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import fixtures from './spatial-lag-maximum-likelihood-operation-fixtures.json';

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
  it(`spatial lag likelihood operations compare every output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = fixtures.find(value => value.name === scene) ?? fixtures[0];
    const tooFew = scene === 'too-few-rows';
    const rowCount = tooFew ? 3 : fixture.response.length;
    const names = [
      'code',
      'converged',
      'table',
      'summary',
      'residuals',
      'powerTraces',
      'powerSums',
      'infinityNorm',
      'errors'
    ];
    const results = [];
    // Each run owns/destroys its operation; always construct a fresh instance.
    for (const Operation of [
      GPUSpatialLagMaximumLikelihoodOperation,
      CPUSpatialLagMaximumLikelihoodOperation
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
                powerTraces: 'powerTraces',
                powerSums: 'powerSums',
                infinityNorm: 'infinityNorm',
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
      // Independent dense NumPy/SciPy literals, generated by the adjacent fixture script.
      // Comparing each backend directly also catches a defect shared by the GPU and CPU twin.
      for (const result of results) {
        for (const port of [
          'table',
          'summary',
          'residuals',
          'powerTraces',
          'powerSums',
          'infinityNorm',
          'powerTraceStandardErrors'
        ] as const) {
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

it('both lag likelihood operations report undefined pseudo-R squared for the reciprocal-pair constant fit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const results = [];
  for (const Operation of [
    GPUSpatialLagMaximumLikelihoodOperation,
    CPUSpatialLagMaximumLikelihoodOperation
  ]) {
    results.push(
      await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8)},
          neighbors: {format: 'uint32', values: Uint32Array.of(4, 5, 6, 7, 0, 1, 2, 3)},
          weights: {format: 'float32', values: new Float32Array(8).fill(1)},
          predictors: {format: 'float32', values: Float32Array.of(0, 0, 0, 0, -1, 1, -1, 1)},
          response: {format: 'float32', values: Float32Array.of(-1, 1, -1, 1, 0, 0, 0, 0)}
        },
        operations: [
          new Operation({
            rowCount: 8,
            predictorCount: 1,
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            predictors: 'predictors',
            response: 'response',
            output: {status: {code: 'code'}, table: 'table', summary: 'summary'}
          })
        ],
        outputs: ['code', 'summary']
      })
    );
  }
  for (const result of results) {
    // Hand derivation: e0=y, e1=0; logdet=4 log(1-rho^2) is maximized at rho=0.
    // Syy=Syu=Suu=4 and Su=0 give prediction covariance=variance=0, hence NaN.
    expect(result['code'][0]).toBe(0);
    expect(Math.abs(result['summary'][1])).toBeLessThan(1e-7);
    expectRegressionRowsClose(
      [result['summary'][0], result['summary'][2], result['summary'][8]],
      [8, 0.5, 4],
      1e-6
    );
    expect(result['summary'][6]).toBeNaN();
  }
  expectRegressionRowsClose([results[0]['summary'][6]], [results[1]['summary'][6]], 0);
});
