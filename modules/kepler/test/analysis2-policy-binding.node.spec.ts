// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  describeNumericNarrowing,
  describeRateInvalidCauses,
  describeResidentGroupByPolicy
} from '../../../examples/deck/kepler-compute/src/analysis2-policy-binding';
import type {ResidentGroupByPlan} from '../../../examples/deck/kepler-compute/src/resident-group-by-plan';

test('numeric narrowing and rate causes describe exact, lossy and excluded rows', () => {
  expect(describeNumericNarrowing('float64', [1, 2.5])).toContain('exactly');
  expect(describeNumericNarrowing('int64', [16777217n])).toContain('rounded in 1 of 1');
  expect(describeNumericNarrowing('float64', [1e300])).toContain('refused');
  expect(describeRateInvalidCauses([1, null, 2], [10, 5, 0])).toBe(
    'Excluded rows: 1 nullEvent, 1 zeroPopulation.'
  );
});

test('group-by policy reports key chunks and measure batches', () => {
  const plan = {keyCapacity: 10, measures: [{}, {}]} as unknown as ResidentGroupByPlan;
  expect(describeResidentGroupByPolicy(plan)).toContain('single GPU pass');
  const wide = {
    keyCapacity: 5000,
    measures: new Array(17).fill({})
  } as unknown as ResidentGroupByPlan;
  expect(describeResidentGroupByPolicy(wide)).toContain('2 GPU passes');
});
