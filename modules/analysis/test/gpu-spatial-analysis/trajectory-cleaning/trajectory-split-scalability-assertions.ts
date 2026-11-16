// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Checks the same independent packed-output expectations in node and WebGPU tests. */
export function expectPackedTrajectorySplitReadback(output: {
  packed: readonly number[];
  offsets: readonly number[];
  rows: readonly number[];
  overflow: readonly number[];
}): void {
  expect(output.packed).toEqual([0, 1]);
  expect(output.offsets).toEqual([0, 2]);
  expect(output.rows).toEqual([0, 1]);
  expect(output.overflow).toEqual([0]);
}

/** Checks saturated prefixes and sentinels using the public number-array readback contract. */
export function expectSaturatedTrajectorySplitReadback(
  output: {
    written: readonly number[];
    prefixes: readonly number[];
    count: readonly number[];
    required: readonly number[];
    overflow: readonly number[];
  },
  segmentOverflow: boolean
): void {
  expect(output.written).toEqual(segmentOverflow ? [99, 99] : [0, 1, 99, 99]);
  expect(output.prefixes).toEqual([0, 0xffffffff]);
  expect(output.count).toEqual([2]);
  expect(output.required).toEqual([0xffffffff]);
  expect(output.overflow).toEqual([1]);
}
