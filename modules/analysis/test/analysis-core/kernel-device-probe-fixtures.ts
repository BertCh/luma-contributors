// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const GRID_QUERY_INITIAL_CANDIDATES = [0xffffffff];

export function createSegmentParityFixture() {
  const values = Array<number>(257).fill(0);
  values[0] = 16777216;
  values[1] = 1;
  values[128] = -16777216;
  values[256] = 2;
  return {
    values: [values[0], 9, values[1], -2, ...values.slice(2)],
    keys: [1, 3, 1, 3, ...Array<number>(255).fill(1)],
    expected: [0, 3, 0, 7]
  };
}

/** Literal 2^24 + 1 - 2^24 cancellation, sum/correction/combined plus a diagnostic word. */
export function validateCompensatedCancellationReadback(values: readonly number[]): void {
  if (
    values.length !== 6 ||
    !values.every(Number.isFinite) ||
    values.slice(0, 4).some((value, index) => value !== [12345, 0, 1, 1][index]) ||
    values[5] !== 12345
  )
    throw new Error('Invalid compensated cancellation readback');
}
