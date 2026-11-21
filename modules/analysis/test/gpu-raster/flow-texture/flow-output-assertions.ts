// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {COMPACT_STATUS_OUTPUTS} from './compact-completeness-fixtures';

export function expectPublicationOrder(nodes: readonly {id: string}[]) {
  const clearIndex = nodes.findIndex(node => node.id === 'publication-clear-state');
  const countsIndex = nodes.findIndex(node => node.id === 'publication-counts');
  expect(clearIndex).toBeGreaterThanOrEqual(0);
  expect(countsIndex).toBeGreaterThanOrEqual(0);
  expect(clearIndex).toBeLessThan(countsIndex);
}

export function expectCompactStatus(
  rows: Readonly<Record<string, ArrayLike<number>>>,
  expected: readonly number[]
) {
  for (const [index, name] of COMPACT_STATUS_OUTPUTS.entries()) {
    expect(Array.from(rows[name])).toEqual([expected[index]]);
  }
}

export function expectLICValues(values: readonly number[], fieldWidth: number) {
  expect(values).toHaveLength(2);
  if (fieldWidth === 1) {
    expect(values).toEqual([0, NaN]);
  } else {
    for (const [index, expected] of [1 / 3, 2 / 3].entries()) {
      expect(Number.isFinite(values[index])).toBe(true);
      expect(values[index]).toBeCloseTo(expected, 6);
    }
  }
}

export function expectLostPosition(values: readonly number[]) {
  expect(values).toEqual([NaN, NaN]);
}
