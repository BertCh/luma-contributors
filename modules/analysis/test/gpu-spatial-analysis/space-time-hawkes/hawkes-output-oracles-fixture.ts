// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {SpaceTimeHawkesOperationProps} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes-operation';
import type {OperationGraphRows} from '../../utils/mixed-execution-test-utils';

export const HAWKES_INPUTS = {
  positions: {format: 'float32x2' as const, values: new Float32Array([4, 0, 0, 0, 0, 0, 2, 0])},
  times: {format: 'float32' as const, values: new Float32Array([2, -1, 0, 1])},
  parameters: {
    format: 'float32' as const,
    values: new Float32Array([1, 4, 10, 0.5, 1e-8, 1, 0.5, 1, 1])
  }
};
const all = {
  candidateOverflow: 'candidateOverflow',
  invalidCount: 'invalidCount',
  approximation: 'approximation'
};
export const HAWKES_STATUS_CASES = [
  all,
  {},
  ...Object.keys(all).flatMap(name => [
    {[name]: name},
    Object.fromEntries(Object.entries(all).filter(([field]) => field !== name))
  ])
];
export function getHawkesProps(
  status: SpaceTimeHawkesOperationProps['output']['status']
): SpaceTimeHawkesOperationProps {
  return {
    rowCount: 4,
    maximumIterations: 1,
    maximumParents: 1,
    positions: 'positions',
    times: 'times',
    parameters: 'parameters',
    output: {estimates: 'estimates', status}
  };
}
export function expectHawkesOracle(
  result: Record<string, OperationGraphRows>,
  gpu: boolean,
  requestedStatus: readonly string[],
  separated = false
): void {
  expect(Object.keys(result).sort()).toEqual(['estimates', ...requestedStatus].sort());
  // Exactly three valid events: t=-1 is outside [0,4]. The t=2 event has two earlier lag
  // candidates, so its capacity-one window is cut once (CPU deliberately has no parent cap).
  // Spatial distances >=2 exceed .5: no triggering, every responsibility is background.
  // One update gives mu=3/(area*duration)=3/4, omega=floor=1e-30, unchanged beta/sigma=1.
  // Finite lag/distance cutoffs intentionally approximate the infinite-support Hawkes kernel.
  expect(result['estimates'][0]).toBeCloseTo(separated ? 0.5 : 0.75, 6);
  expect(result['estimates'][1]).toBe(Math.fround(1e-30));
  expect(Array.from(result['estimates'].slice(2))).toEqual([1, 1]);
  if (result['candidateOverflow'])
    expect(Array.from(result['candidateOverflow'])).toEqual([gpu && !separated ? 1 : 0]);
  if (result['invalidCount'])
    expect(Array.from(result['invalidCount'])).toEqual([separated ? 2 : 1]);
  if (result['approximation']) expect(Array.from(result['approximation'])).toEqual([1]);
}

// Two excluded rows and at most one earlier candidate: distinct diagnostics [0,2,1], mu=2/4.
export const SEPARATED_HAWKES_INPUTS = {
  ...HAWKES_INPUTS,
  times: {format: 'float32' as const, values: new Float32Array([-1, -2, 0, 1])}
};
