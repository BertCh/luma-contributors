// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  diagnoseFloat32Narrowing,
  mergeNumericNarrowingReports,
  summarizeNarrowingReference,
  summarizeInteger64Reference,
  summarizeDecimal128Reference,
  normalizeDecimal128Comparison,
  GPUWideNumericScan,
  type GPUWideNumericScanProps
} from '../../../src/dataframe-analysis/column-profile/index';

test('integer maximum error subtracts exact source values before converting to Number', () => {
  for (const [format, values, maximum] of [
    ['uint64', [(1n << 64n) - 1n, (1n << 63n) + 1n, null], 1n],
    ['int64', [-(1n << 63n) + 1n, (1n << 63n) - 1n], 1n],
    // This lies one integer above a halfway boundary but Number() rounds it back to the tie.
    ['uint64', [(1n << 63n) + (1n << 39n) + 1n], (1n << 39n) - 1n]
  ] as const) {
    const report = diagnoseFloat32Narrowing(format, values);
    expect(report.maximumAbsoluteError).toBe(maximum);
    expect(report.inexactCount).toBe(values.filter(value => value !== null).length);
    expect(report).toEqual(summarizeNarrowingReference(format, values));
  }
  const empty = diagnoseFloat32Narrowing('int64', []);
  expect(empty.maximumAbsoluteError).toBe(0n);
  expect(
    mergeNumericNarrowingReports([empty, diagnoseFloat32Narrowing('int64', [16777217n])], 'int64')
      .maximumAbsoluteError
  ).toBe(1n);
});

test('Float64 error and overflow remain distinct from exact integer errors', () => {
  const report = diagnoseFloat32Narrowing('float64', [1, 16777217, null, Infinity, 1e300]);
  expect(report).toMatchObject({
    maximumAbsoluteError: 1,
    nullCount: 1,
    inexactCount: 1,
    nonFiniteCount: 1,
    overflowCount: 1,
    verdict: 'refused'
  });
});

test('CPU exact summaries retain sums beyond the input word width and decimal scale', () => {
  const maximum = (1n << 63n) - 1n;
  expect(summarizeInteger64Reference([maximum, maximum, null, -1n], true)).toEqual({
    count: 3,
    nullCount: 1,
    minimum: -1n,
    maximum,
    sum: (1n << 64n) - 3n
  });
  const decimalMaximum = (1n << 127n) - 1n;
  expect(summarizeDecimal128Reference([decimalMaximum, decimalMaximum, null, -1n], 3)).toEqual({
    count: 3,
    nullCount: 1,
    minimum: -1n,
    maximum: decimalMaximum,
    sum: (1n << 128n) - 3n,
    scale: 3
  });
  expect(normalizeDecimal128Comparison('>=', 12345n, 3, 2)).toEqual({
    kind: 'compare',
    operator: '>=',
    threshold: 1235n
  });
});

test('wide scans validate layout, exact counter limit and node budget at construction', () => {
  const chunk = {length: 2, byteStride: 8, byteOffset: 0};
  const props = {
    id: 'wide-test',
    words: {format: 'uint32x2', length: 2, data: [chunk]},
    output: {format: 'uint32', byteOffset: 0, length: 6, byteStride: 4},
    resultWordCount: 6,
    phaseBodies: ['', '']
  } as unknown as GPUWideNumericScanProps;
  expect(new GPUWideNumericScan(props).estimatedCommandNodeCount).toBe(2);
  expect(
    () =>
      new GPUWideNumericScan({
        ...props,
        words: {...props.words, length: 2 ** 32} as typeof props.words
      })
  ).toThrow(/wide-test rowCount.*4294967295/);
  expect(() => new GPUWideNumericScan({...props, phaseBodies: Array(32769).fill('')})).toThrow(
    /wide-test estimated command node count/
  );
  expect(
    () =>
      new GPUWideNumericScan({
        ...props,
        validity: {data: []} as unknown as NonNullable<GPUWideNumericScanProps['validity']>
      })
  ).toThrow(/wide-test validity/);
});
