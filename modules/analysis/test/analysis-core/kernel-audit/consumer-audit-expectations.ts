// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Shared device/readback oracle for the matched key 7 and unmatched key 8. */
export function expectJoinedFloatAggregates(
  sums: readonly number[],
  means: readonly number[]
): void {
  expect(sums).toHaveLength(2);
  expect(means).toHaveLength(2);
  expect(Number.isFinite(sums[0])).toBe(true);
  expect(sums[0]).toBe(3);
  expect(Number.isNaN(sums[1])).toBe(true);
  expect(Number.isFinite(means[0])).toBe(true);
  expect(means[0]).toBe(1.5);
  expect(Number.isNaN(means[1])).toBe(true);
}

export type NearestTieReadback = {
  ties: 'lowest-row' | 'all';
  capacity: number;
  count: number;
  rows: readonly number[];
  ids: readonly number[];
  distances: readonly number[];
  overflow: number;
};

/** Equal-distance features on rows [0,1] have application IDs [99,1]. */
export function expectNearestTieResult(result: NearestTieReadback): void {
  expect(result.overflow).toBe(result.ties === 'all' && result.capacity === 1 ? 1 : 0);
  const both = result.ties === 'all' && result.capacity > 1;
  expect(result.count).toBe(both ? 2 : 1);
  expect(result.rows).toEqual(both ? [0, 1] : [0]);
  expect(result.ids).toEqual(both ? [99, 1] : [99]);
  expect(result.distances.every(Number.isFinite)).toBe(true);
  expect(result.distances).toEqual(both ? [1, 1] : [1]);
}
