// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {ContinuousSpatialSampleOperationProps} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample-operation';
import type {OperationGraphRows} from '../../utils/mixed-execution-test-utils';

export const CONTINUOUS_PROPS: ContinuousSpatialSampleOperationProps = {
  frameRowCount: 2,
  frame: {type: 'line', positions: 'positions'},
  spacing: 0.5,
  seed: 1,
  capacity: 5,
  output: {
    positions: 'selected',
    designWeights: 'weights',
    provenance: 'provenance',
    status: 'status'
  }
};
export const CONTINUOUS_INPUTS = {
  positions: {format: 'float32x2' as const, values: new Float32Array([0, 2, 2, 2])}
};

export function expectContinuousOracle(result: Record<string, OperationGraphRows>): void {
  // Generated independently by continuous-sample-output-oracles.oracle.py; all four sites plus
  // one unused slot. 2e-6 allows a few float32 interpolation/addition roundings on a length-2 line.
  const positions = [
    0.3629690706729889, 2, 0.8629690408706665, 2, 1.3629690408706665, 2, 1.8629690408706665, 2, 0, 0
  ];
  expect(result['selected'].length).toBe(10);
  for (const [index, expected] of positions.entries()) {
    expect(Number.isFinite(result['selected'][index])).toBe(true);
    expect(Math.abs(result['selected'][index] - expected)).toBeLessThanOrEqual(2e-6);
  }
  expect(Array.from(result['weights'])).toEqual([0.5, 0.5, 0.5, 0.5, 0]);
  // All sites belong to source segment 0, replicate 0, ordinals 0..3, primary role 0.
  expect(Array.from(result['provenance'])).toEqual([
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0xffffffff, 0, 0xffffffff, 0xffffffff
  ]);
  expect(Array.from(result['status'])).toEqual([4, 4, 4, 0]);
}
