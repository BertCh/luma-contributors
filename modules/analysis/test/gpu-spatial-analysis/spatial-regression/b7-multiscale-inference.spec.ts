// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMultiscaleGeographicallyWeightedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation';
import {GPUMultiscaleGeographicallyWeightedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createMultiscaleOverflowFixture} from './multiscale-gwr-audit-fixtures';

it.each([
  [2, 5.333333333333333, 0.816496580927726, 0.816496580927726],
  [1e19, 1.3333333281350528e38, 4.082482896680421e18, 4.082482896680421e18],
  [2 ** 63, 2 ** 127, 6.521908801048674e18, 5.2175270408389394e19]
])('R6a MGWR publishes finite full inference at response magnitude %s', async (magnitude, variance, interceptError, slopeError) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createMultiscaleOverflowFixture();
  const fourRows = magnitude === 2 ** 63;
  const response = fourRows
    ? Float32Array.of(magnitude, -magnitude, magnitude, -magnitude)
    : fixture.response.map(value => Math.sign(value) * magnitude);
  const predictors = fourRows ? Float32Array.of(1 / 8, 1 / 8, -1 / 8, -1 / 8) : fixture.predictors;
  const rowCount = response.length;
  // Coincident sites give unit weights and a rank-two projector. The response is orthogonal
  // to both design columns. /usr/bin/python3: variance = rowCount * storedResponse**2 / (rowCount-2).
  // Errors are sqrt(variance / sum(column**2)). The four-row predictor has squared norm 1/16.
  for (const Operation of [
    GPUMultiscaleGeographicallyWeightedRegressionOperation,
    CPUMultiscaleGeographicallyWeightedRegressionOperation
  ]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: fixture.positions.subarray(0, 2 * rowCount)},
        predictors: {format: 'float32', values: predictors},
        response: {format: 'float32', values: response},
        parameters: {format: 'float32', values: fixture.parameters}
      },
      operations: [
        new Operation({
          rowCount,
          predictorCount: 1,
          maximumIterations: 4,
          positions: 'positions',
          predictors: 'predictors',
          response: 'response',
          parameters: 'parameters',
          inference: {
            output: {
              standardErrors: 'errors',
              effectiveParameters: 'effective',
              summary: 'summary',
              status: 'inferenceStatus'
            }
          },
          output: {
            coefficients: 'coefficients',
            fitted: 'fitted',
            status: {codes: 'codes', iterationCount: 'iterations'}
          }
        })
      ],
      outputs: ['fitted', 'codes', 'errors', 'effective', 'summary', 'inferenceStatus']
    });
    expect(Array.from(result['fitted'])).toEqual(Array(rowCount).fill(0));
    expect(Array.from(result['codes'])).toEqual(Array(rowCount).fill(0));
    expect(Array.from(result['inferenceStatus'])).toEqual([0]);
    expect(Array.from(result['effective'])).toEqual([1, 1]);
    expect(result['summary']).toHaveLength(4);
    const [trace, traceSquared, actualVariance, degrees] = result['summary'];
    expect([trace, traceSquared, degrees]).toEqual([2, 2, rowCount - 2]);
    expect(Number.isFinite(actualVariance)).toBe(true);
    expect(Math.abs(actualVariance / variance - 1)).toBeLessThan(2e-6);
    expect(result['errors']).toHaveLength(2 * rowCount);
    result['errors'].forEach((error, index) => {
      const standardError = index % 2 === 0 ? interceptError : slopeError;
      expect(Number.isFinite(error)).toBe(true);
      expect(Math.abs(error / standardError - 1)).toBeLessThan(2e-6);
    });
  }
});
