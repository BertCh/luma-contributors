// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {compareCellTablesOnCPU, convertInt64ToFloat32} from './cell-table-compare-oracle';

it('rounds full-width differences once using guard and sticky bits from both words', () => {
  const before = -(1n << 62n);
  const after = (1n << 62n) + (1n << 39n) + (1n << 31n);
  const result = compareCellTablesOnCPU(
    [
      {key: 1n, count: 1, sum: before},
      {key: 2n, count: 1, sum: after}
    ],
    [
      {key: 1n, count: 1, sum: after},
      {key: 2n, count: 1, sum: before}
    ],
    {measure: 'sum', sumScale: 1, zScore: 'poisson', capacity: 2}
  );
  expect(result.rows.map(row => row.delta)).toEqual([2 ** 63 + 2 ** 40, -(2 ** 63 + 2 ** 40)]);
  const midpoint = (1n << 63n) + (1n << 39n);
  expect(convertInt64ToFloat32(midpoint - 1n)).toBe(2 ** 63);
  expect(convertInt64ToFloat32(midpoint)).toBe(2 ** 63);
  expect(convertInt64ToFloat32(midpoint + 1n)).toBe(2 ** 63 + 2 ** 40);
  expect(convertInt64ToFloat32(midpoint + (1n << 40n))).toBe(2 ** 63 + 2 ** 41);
});

it('standardizes full-width changes without overflowing at either sum scale', () => {
  const minimum = -(1n << 63n);
  const maximum = (1n << 63n) - 1n;
  for (const sumScale of [1, 65536]) {
    const result = compareCellTablesOnCPU(
      [
        {key: 1n, count: 1, sum: minimum},
        {key: 2n, count: 1, sum: maximum}
      ],
      [
        {key: 1n, count: 1, sum: maximum},
        {key: 2n, count: 1, sum: minimum}
      ],
      {measure: 'sum', sumScale, zScore: 'standardized', capacity: 2}
    );
    expect(result.rows.map(row => row.zScore)).toEqual([1, -1]);
  }
});

it('keeps finite full-width percent changes at a tiny sum scale', () => {
  const result = compareCellTablesOnCPU(
    [{key: 1n, count: 1, sum: -(1n << 62n)}],
    [{key: 1n, count: 1, sum: 1n << 62n}],
    {measure: 'sum', sumScale: 2 ** -60, zScore: 'poisson', capacity: 1}
  );
  expect(Number.isFinite(result.rows[0].before)).toBe(true);
  expect(Number.isFinite(result.rows[0].after)).toBe(true);
  expect(Number.isFinite(result.rows[0].delta)).toBe(true);
  expect(result.rows[0].percentChange).toBe(-200);
});

it('cell-table differences preserve the sign beyond signed 64-bit range and near cancellation', () => {
  // Each input fits signed int64; their difference needs a sign plus a full uint64 magnitude.
  const cases = [
    {before: -(1n << 62n), after: 1n << 62n, expected: 2 ** 63},
    {before: 1n << 62n, after: -(1n << 62n), expected: -(2 ** 63)},
    {before: (1n << 63n) - 1n, after: -(1n << 63n), expected: -(2 ** 64)},
    {before: -(1n << 63n), after: (1n << 63n) - 1n, expected: 2 ** 64},
    {before: (1n << 62n) + 17n, after: (1n << 62n) + 20n, expected: 3},
    {before: -(1n << 62n) - 17n, after: -(1n << 62n) - 20n, expected: -3}
  ];
  const result = compareCellTablesOnCPU(
    cases.map(({before}, row) => ({key: BigInt(row + 1), count: 1, sum: before})),
    cases.map(({after}, row) => ({key: BigInt(row + 1), count: 1, sum: after})),
    {measure: 'sum', sumScale: 1, zScore: 'poisson', capacity: cases.length}
  );
  expect(result.overflow).toBe(false);
  expect(result.rows.map(row => row.delta)).toEqual(cases.map(entry => entry.expected));
});
