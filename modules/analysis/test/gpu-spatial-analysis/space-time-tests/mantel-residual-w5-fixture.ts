// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {getGPUSpaceTimeParameterValues} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import type {MantelTestOperationProps} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation';

export function getResidualInputs(star: boolean, seed: number) {
  return {
    offsets: {
      format: 'uint32' as const,
      values: Uint32Array.from(star ? [0, 3, 3, 3, 3] : [0, 2, 3, 3])
    },
    neighbors: {format: 'uint32' as const, values: Uint32Array.from(star ? [1, 2, 3] : [1, 2, 2])},
    weights: {format: 'float32' as const, values: Float32Array.of(1, 1, 1)},
    distances: {
      format: 'float32' as const,
      values: Float32Array.from(star ? [2, 1, 2] : [1, 10001, 10000])
    },
    times: {
      format: 'float32' as const,
      values: Float32Array.from(star ? [0, 2, 11, 20] : [0, 1, 10001])
    },
    parameters: {
      format: 'uint32' as const,
      values: getGPUSpaceTimeParameterValues({seed, permutationCount: 2})
    }
  };
}

export function getResidualProps(star: boolean, stopping = false): MantelTestOperationProps {
  return {
    rowCount: star ? 4 : 3,
    pairs: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights', distances: 'distances'},
    times: 'times',
    parameters: 'parameters',
    maximumPermutations: 2,
    ...(stopping ? {stopAfterExceedances: 1} : {}),
    output: {
      statistics: 'statistics',
      summary: 'summary',
      permutationCount: 'used',
      usablePermutationCount: 'usable'
    }
  };
}

export function expectResidualInference(
  result: Record<string, ArrayLike<number>>,
  star: boolean,
  stopping: boolean
): void {
  // Integer Pearson numerator n*sum(x*y)-sum(x)*sum(y) is exactly zero for
  // the star's observed/reversed gaps [2,11,20]/[20,11,2]. Seed84's two nulls
  // have this same covariance. For the triangle, seed8's nulls are the observed
  // gap order [1,10001,10000] and [1,10000,10001]. The centered integer variance
  // numerator is 199980002; the correlation difference is 3/199980002. Python
  // Decimal(60) evaluation gives sample variance 9/(2*199980002^2)=1.1252250112477496e-16,
  // z=1/sqrt(2), and one (not two) upper exceedance.
  expect(Array.from(result['used'])).toEqual([stopping ? 1 : 2]);
  expect(Array.from(result['usable'])).toEqual([stopping ? 1 : 2]);
  if (star) {
    expect(result['statistics'][0]).toBe(0);
    expect(result['summary'][5]).toBe(0);
    expect(Number.isNaN(result['summary'][11])).toBe(true);
  } else {
    expect(result['summary'][6]).toBe(1);
    expect(Number.isFinite(result['summary'][5])).toBe(true);
    expect(Math.abs(result['summary'][5] / 1.1252250112477496e-16 - 1)).toBeLessThan(0.001);
    expect(Number.isFinite(result['summary'][11])).toBe(true);
    expect(Math.abs(result['summary'][11] - Math.SQRT1_2)).toBeLessThan(0.001);
  }
}
