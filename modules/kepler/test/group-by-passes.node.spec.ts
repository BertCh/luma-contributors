// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  mergeKeplerGroupByPasses,
  planKeplerGroupByPasses,
  type KeplerGroupByPass,
  type KeplerGroupByPassResult
} from '../src/aggregation/kepler-group-by-passes';

/** CPU model of one GPU pass: groups in window-local codes, ordered by first source row. */
function simulatePass(
  pass: KeplerGroupByPass,
  slots: readonly number[],
  values: readonly number[],
  operations: readonly ('count' | 'sum')[]
): KeplerGroupByPassResult {
  const groups = new Map<number, {first: number; rows: number[]}>();
  slots.forEach((slot, row) => {
    if (slot < pass.keyStart || slot >= pass.keyStart + pass.keyCapacity) return;
    const group = groups.get(slot - pass.keyStart) ?? {first: row, rows: []};
    group.rows.push(row);
    groups.set(slot - pass.keyStart, group);
  });
  const ordered = [...groups.entries()].sort((a, b) => a[1].first - b[1].first);
  const batch = operations.slice(pass.measureStart, pass.measureEnd);
  return {
    pass,
    keys: Uint32Array.from(ordered.map(([key]) => key)),
    firstRows: Uint32Array.from(ordered.map(([, group]) => group.first)),
    values: batch.map(operation =>
      operation === 'count'
        ? Uint32Array.from(ordered.map(([, group]) => group.rows.length))
        : Float32Array.from(
            ordered.map(([, group]) => group.rows.reduce((total, row) => total + values[row], 0))
          )
    ),
    validity: batch.map(() => Uint32Array.from(ordered.map(() => 1)))
  };
}

test('pass planning keeps small group-bys in one pass and splits wide ones', () => {
  const small = planKeplerGroupByPasses({keySlotCount: 100, measureCount: 3, rowCount: 1000});
  expect(small.passes).toHaveLength(1);
  const wide = planKeplerGroupByPasses({keySlotCount: 100, measureCount: 40, rowCount: 1000});
  expect(wide.batchCount).toBe(3);
  expect(wide.passes.map(pass => [pass.measureStart, pass.measureEnd])).toEqual([
    [0, 16],
    [16, 32],
    [32, 40]
  ]);
  const keys = planKeplerGroupByPasses({
    keySlotCount: 10_000,
    measureCount: 2,
    rowCount: 1000,
    maximumSlotsPerPass: 4096
  });
  expect(keys.windowCount).toBe(3);
  expect(keys.passes.map(pass => pass.keyCapacity)).toEqual([4096, 4096, 1808]);
  const large = planKeplerGroupByPasses({
    keySlotCount: 4_000_000,
    measureCount: 16,
    rowCount: 60_000_000
  });
  expect(large.passes.reduce((total, pass) => total + pass.keyCapacity, 0)).toBe(
    4_000_000 * large.batchCount
  );
});

test('merging key windows and measure batches reproduces first-occurrence order', () => {
  const slots = [5, 2, 5, 9, 0, 2, 9, 7, 0, 5];
  const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const operations = ['count', 'sum', 'sum'] as const;
  const plan = planKeplerGroupByPasses({
    keySlotCount: 10,
    measureCount: 3,
    rowCount: slots.length,
    maximumSlotsPerPass: 4,
    maximumMeasuresPerPass: 2
  });
  expect(plan.passes.length).toBe(plan.windowCount * plan.batchCount);
  // The planner never shrinks windows below 4096 slots; model smaller windows explicitly.
  const passes: KeplerGroupByPass[] = [];
  for (const keyStart of [0, 4, 8])
    for (const [measureStart, measureEnd] of [
      [0, 2],
      [2, 3]
    ])
      passes.push({keyStart, keyCapacity: Math.min(4, 10 - keyStart), measureStart, measureEnd});
  const merged = mergeKeplerGroupByPasses(
    passes.map(pass => simulatePass(pass, slots, values, operations)),
    3,
    operations.map(operation => operation === 'count')
  );
  expect([...merged.slots]).toEqual([5, 2, 9, 0, 7]);
  expect([...merged.firstRows]).toEqual([0, 1, 3, 4, 7]);
  expect([...merged.values[0]]).toEqual([3, 2, 2, 2, 1]);
  expect([...merged.values[1]]).toEqual([14, 8, 11, 14, 8]);
  expect([...merged.values[2]]).toEqual([14, 8, 11, 14, 8]);
  const single = mergeKeplerGroupByPasses(
    [
      simulatePass(
        {keyStart: 0, keyCapacity: 10, measureStart: 0, measureEnd: 3},
        slots,
        values,
        operations
      )
    ],
    3,
    operations.map(operation => operation === 'count')
  );
  expect([...single.slots]).toEqual([...merged.slots]);
  expect([...single.values[1]]).toEqual([...merged.values[1]]);
});

test('merging rejects batches that disagree on group identity', () => {
  const pass = (measureStart: number): KeplerGroupByPass => ({
    keyStart: 0,
    keyCapacity: 4,
    measureStart,
    measureEnd: measureStart + 1
  });
  const result = (measureStart: number, firstRows: number[]): KeplerGroupByPassResult => ({
    pass: pass(measureStart),
    keys: Uint32Array.from([0, 1]),
    firstRows: Uint32Array.from(firstRows),
    values: [Uint32Array.from([1, 1])],
    validity: [Uint32Array.from([1, 1])]
  });
  expect(() =>
    mergeKeplerGroupByPasses([result(0, [0, 1]), result(1, [0, 2])], 2, [true, true])
  ).toThrow(/disagree/);
});
