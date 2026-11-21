// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {SpatialBalancedSampleOperationProps} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample-operation';
import type {OperationGraphRows} from '../../utils/mixed-execution-test-utils';

export const BALANCED_PROPS: SpatialBalancedSampleOperationProps = {
  rowCount: 2,
  positions: 'positions',
  frameIds: 'ids',
  strata: 'strata',
  bounds: [0, 0, 1, 1],
  seed: 79,
  allocations: [
    {stratumId: 9, sampleCount: 1},
    {stratumId: 3, sampleCount: 1}
  ],
  output: {
    frameIds: 'selected',
    inclusionProbabilities: 'probabilities',
    designWeights: 'weights',
    status: 'status'
  }
};
export function getBalancedInputs(invalid = false) {
  return {
    positions: {
      format: 'float32x2' as const,
      values: new Float32Array([0.25, 0.25, invalid ? NaN : 0.75, 0.75])
    },
    ids: {format: 'uint32' as const, values: new Uint32Array([71, 29])},
    strata: {format: 'uint32' as const, values: new Uint32Array([3, 9])}
  };
}
export function expectBalancedOracle(result: Record<string, OperationGraphRows>, invalid = false) {
  // Each stratum contains one member selected with certainty, independent of every random word.
  // Allocation order reverses input row order; caller IDs must survive verbatim.
  expect(Array.from(result['selected'])).toEqual(invalid ? [0xffffffff, 0xffffffff] : [29, 71]);
  expect(Array.from(result['status'])).toEqual(invalid ? [0, 1, 0, 1] : [1, 0, 1, 0]);
  for (const name of ['probabilities', 'weights']) {
    if (invalid) for (const value of result[name]) expect(Number.isNaN(value)).toBe(true);
    else expect(Array.from(result[name])).toEqual([1, 1]);
  }
}

export const FRACTIONAL_BALANCED_PROPS = {...BALANCED_PROPS, rowCount: 3};
export const FRACTIONAL_BALANCED_INPUTS = {
  positions: {
    format: 'float32x2' as const,
    values: new Float32Array([0.25, 0.25, 0.75, 0.75, 0.25, 0.25])
  },
  ids: {format: 'uint32' as const, values: new Uint32Array([71, 29, 83])},
  strata: {format: 'uint32' as const, values: new Uint32Array([3, 9, 3])}
};
export function expectFractionalBalancedOracle(result: Record<string, OperationGraphRows>): void {
  // Independent Random123 oracle: shift=1 in stable tied order [71,83]. n/N=1/2, inverse=2.
  expect(Object.keys(result).sort()).toEqual(
    ['selected', 'probabilities', 'weights', 'status'].sort()
  );
  expect(Array.from(result['selected'])).toEqual([29, 83]);
  expect(Array.from(result['probabilities'])).toEqual([1, 0.5]);
  expect(Array.from(result['weights'])).toEqual([1, 2]);
  expect(Array.from(result['status'])).toEqual([1, 0, 1, 0]);
}
