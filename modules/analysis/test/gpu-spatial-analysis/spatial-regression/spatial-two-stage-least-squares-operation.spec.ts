// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {GPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import fixtures from './spatial-two-stage-least-squares-operation-fixtures.json';

for (const scene of ['illConditioned', 'singularRobust', 'insufficient'] as const) {
  it(`checks the review's GPU hypothesis: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const insufficient = scene === 'insufficient';
    const epsilon = 2 ** -24;
    const predictors = insufficient
      ? Float32Array.of(-1, 0, 1)
      : scene === 'singularRobust'
        ? new Float32Array(8)
        : Float32Array.of(-1, -1 + epsilon, 1, 1, -1, -1, 1, 1 - epsilon);
    const response = insufficient
      ? Float32Array.of(-1, 0, 1)
      : Float32Array.of(1, 3, -1, 2, 7, 0, -2, 8);
    const neighbors = insufficient
      ? Uint32Array.of(1, 2, 0)
      : Uint32Array.of(1, 0, 3, 2, 5, 4, 7, 6);
    const rowCount = response.length;
    const robust = scene === 'singularRobust';
    const outputNames = [
      'code',
      'table',
      'summary',
      'residuals',
      ...(robust ? ['covariance', 'robustStatus'] : [])
    ];
    const results = [];
    for (const Operation of [
      GPUSpatialTwoStageLeastSquaresOperation,
      CPUSpatialTwoStageLeastSquaresOperation
    ]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {
            predictors: {format: 'float32', values: predictors},
            response: {format: 'float32', values: response},
            offsets: {
              format: 'uint32',
              values: Uint32Array.from({length: rowCount + 1}, (_, row) => row)
            },
            neighbors: {format: 'uint32', values: neighbors},
            weights: {format: 'float32', values: new Float32Array(rowCount).fill(1)}
          },
          operations: [
            new Operation({
              rowCount,
              predictorCount: 1,
              predictors: 'predictors',
              response: 'response',
              weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
              robust: robust
                ? {method: 'HC0', output: {covariance: 'covariance', status: 'robustStatus'}}
                : undefined,
              output: {
                status: {code: 'code'},
                table: 'table',
                summary: 'summary',
                residuals: 'residuals'
              }
            })
          ],
          outputs: outputNames
        })
      );
    }
    const [gpu, cpu] = results;
    if (scene === 'illConditioned') {
      // The two orthogonal directions (x+Wx)/2 and (x-Wx)/2 give the exact slope
      // -1801439877791744/1801439810682881, rounded to -1. The CPU must recover it.
      expect(cpu['code'][0]).toBe(0);
      expect(cpu['table'][0]).toBe(4.5);
      expect(cpu['table'][4]).toBe(-1);
      expect(cpu['table'][8]).toBe(-1);
      // The instrument correlation condition number is about 2.25e15: float32 normal equations
      // cannot resolve its small direction. The integration GPU accepted this scene with status
      // 0 and an intercept 2.63% from the exact IV result. Status 1 is also permitted by the
      // documented rank difference. Ordinary accuracy bounds do not apply here; keep the exact
      // CPU oracle above and check the GPU's precision warning and published model identities.
      expect([0, 1], `GPU status=${gpu['code'][0]}, slope=${gpu['table'][4]}`).toContain(
        gpu['code'][0]
      );
      if (gpu['code'][0] === 0) {
        expect(gpu['summary'][0]).toBe(rowCount);
        expect(gpu['summary'][10]).toBe(1);
        expect(gpu['summary'][13]).toBe(1);
        expect(gpu['table'].every(Number.isFinite)).toBe(true);
        expect(gpu['residuals'].every(Number.isFinite)).toBe(true);
        // These bounds check publication/reduction rounding, not proximity to the CPU estimate:
        // 32 float32 unit roundoffs relative to the sum of the structural terms, or to RSS.
        const publicationRoundoff = 32 * 2 ** -24;
        let residualSquares = 0;
        for (let row = 0; row < rowCount; row++) {
          const intercept = gpu['table'][0];
          const predictorTerm = predictors[row] * gpu['table'][4];
          const lagTerm = response[neighbors[row]] * gpu['table'][8];
          const residual = gpu['residuals'][row];
          const magnitude =
            Math.abs(response[row]) +
            Math.abs(intercept) +
            Math.abs(predictorTerm) +
            Math.abs(lagTerm);
          expect(
            Math.abs(residual - (response[row] - intercept - predictorTerm - lagTerm))
          ).toBeLessThanOrEqual(publicationRoundoff * magnitude);
          residualSquares += residual ** 2;
        }
        expect(residualSquares).toBeGreaterThan(0);
        expect(Math.abs(gpu['summary'][2] - residualSquares)).toBeLessThanOrEqual(
          publicationRoundoff * residualSquares
        );
        expect(Math.abs(gpu['summary'][1] - residualSquares / rowCount)).toBeLessThanOrEqual(
          (publicationRoundoff * residualSquares) / rowCount
        );
      } else {
        expect(gpu['table'].every(Number.isNaN)).toBe(true);
        expect(gpu['residuals'].every(Number.isNaN)).toBe(true);
        expect(gpu['summary'][0]).toBe(8);
        expect(gpu['summary'].subarray(1).every(Number.isNaN)).toBe(true);
      }
    } else if (scene === 'singularRobust') {
      // Both instrument columns have zero diagonal: finite rank failure (1), not non-finite (3).
      for (const result of results) {
        expect(result['code'][0]).toBe(1);
        expect(result['robustStatus'][0]).toBe(1);
      }
      for (const name of outputNames) expectRegressionRowsClose(gpu[name], cpu[name], 0);
    } else {
      // Saturated IV fit beta=1, rho=0 has algebraic residuals zero, but n=3 requires status 2.
      // GPU publishes residuals before fail(2); the CPU's documented failure policy clears them.
      for (const name of ['code', 'table', 'summary'])
        expectRegressionRowsClose(gpu[name], cpu[name], 0);
      expect(cpu['code'][0]).toBe(2);
      expect(cpu['residuals'].every(Number.isNaN)).toBe(true);
      expect(gpu['residuals'].every(Number.isFinite)).toBe(true);
      for (const value of gpu['residuals']) expect(Math.abs(value)).toBeLessThan(1e-5);
    }
  });
}

for (const scene of [
  'ordinary',
  'order2',
  'HC0',
  'HC1',
  'HC2',
  'HC3',
  'CLUSTER',
  'oneGroup',
  'missingKey',
  'nonPSD',
  'HAC',
  'zeroTotal',
  'singular',
  'invalidKernel'
] as const) {
  it(`2SLS operations agree on every output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = fixtures[scene === 'zeroTotal' ? 2 : scene === 'order2' ? 1 : 0];
    const robust = !['ordinary', 'order2', 'zeroTotal', 'singular'].includes(scene);
    const method =
      scene === 'HC0' || scene === 'HC1' || scene === 'HC2' || scene === 'HC3'
        ? scene
        : ['CLUSTER', 'oneGroup', 'missingKey'].includes(scene)
          ? 'CLUSTER'
          : 'HAC';
    const kernel = scene === 'nonPSD' ? fixture.nonPositiveSemidefiniteKernel : fixture.kernel;
    const outputNames = [
      'code',
      'table',
      'summary',
      'residuals',
      ...(robust ? ['covariance', 'robustStatus'] : [])
    ];
    const results = [];
    for (const Operation of [
      GPUSpatialTwoStageLeastSquaresOperation,
      CPUSpatialTwoStageLeastSquaresOperation
    ]) {
      results.push(
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
            offsets: {format: 'uint32', values: Uint32Array.from(fixture.weights.offsets)},
            neighbors: {format: 'uint32', values: Uint32Array.from(fixture.weights.neighbors)},
            weights: {format: 'float32', values: Float32Array.from(fixture.weights.weights)},
            ...(method === 'CLUSTER'
              ? {
                  clusters: {
                    format: 'uint32' as const,
                    values: Uint32Array.from(fixture.clusters, (key, row) =>
                      scene === 'oneGroup'
                        ? 0
                        : scene === 'missingKey' && row === 0
                          ? 0xffffffff
                          : key
                    )
                  }
                }
              : {}),
            ...(robust && method === 'HAC'
              ? {
                  kernelOffsets: {
                    format: 'uint32' as const,
                    values: Uint32Array.from(kernel.offsets)
                  },
                  kernelNeighbors: {
                    format: 'uint32' as const,
                    values: Uint32Array.from(kernel.neighbors)
                  },
                  kernelWeights: {
                    format: 'float32' as const,
                    values:
                      scene === 'invalidKernel'
                        ? new Float32Array(kernel.weights.length)
                        : Float32Array.from(kernel.weights)
                  }
                }
              : {})
          },
          operations: [
            new Operation({
              rowCount: fixture.response.length,
              predictorCount: fixture.predictorCount,
              instrumentOrder: fixture.instrumentOrder as 1 | 2,
              predictors: 'predictors',
              response: 'response',
              weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
              robust: robust
                ? {
                    method,
                    clusters: method === 'CLUSTER' ? 'clusters' : undefined,
                    clusterCount: method === 'CLUSTER' ? fixture.clusterCount : undefined,
                    kernel:
                      method === 'HAC'
                        ? {
                            offsets: 'kernelOffsets',
                            neighbors: 'kernelNeighbors',
                            weights: 'kernelWeights'
                          }
                        : undefined,
                    output: {covariance: 'covariance', status: 'robustStatus'}
                  }
                : undefined,
              output: {
                status: {code: 'code'},
                table: 'table',
                summary: 'summary',
                residuals: 'residuals'
              }
            })
          ],
          outputs: outputNames
        })
      );
    }
    expect(Array.from(results[0]['code'])).toEqual(Array.from(results[1]['code']));
    expect(results[1]['code'][0]).toBe(scene === 'singular' ? 1 : 0);
    if (robust) {
      expect(Array.from(results[0]['robustStatus'])).toEqual(
        Array.from(results[1]['robustStatus'])
      );
      expect(results[1]['robustStatus'][0]).toBe(
        scene === 'invalidKernel'
          ? 4
          : scene === 'nonPSD'
            ? 7
            : scene === 'oneGroup' || scene === 'missingKey'
              ? 6
              : 0
      );
    }
    for (const name of outputNames.filter(name => name !== 'code' && name !== 'robustStatus')) {
      // Relative tolerances: diagnostics and robust normal-equation projections dominate error.
      expectRegressionRowsClose(
        results[0][name],
        results[1][name],
        name === 'summary' || name === 'covariance' ? 5e-3 : 5e-4
      );
    }
    if (scene === 'zeroTotal') {
      expect(results[0]['summary'][4]).toBeNaN();
      expect(results[0]['summary'][5]).toBeCloseTo(0.9505670553, 4);
    }
  });
}
