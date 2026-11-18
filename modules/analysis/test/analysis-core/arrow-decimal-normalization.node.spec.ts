// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {prepareArrowTableForGPUAnalytics} from '../../src/analysis-core/graph-support/arrow-analytics-normalization';

function makeDecimalTable(coefficient: bigint, scale: number, bitWidth = 128) {
  const type = new arrow.Decimal(
    scale,
    bitWidth === 64 ? 18 : bitWidth === 128 ? 38 : 76,
    bitWidth
  );
  const bits = BigInt.asUintN(bitWidth, coefficient);
  const data = Uint32Array.from({length: bitWidth / 32}, (_, wordIndex) =>
    Number((bits >> BigInt(wordIndex * 32)) & 0xffffffffn)
  );
  return new arrow.Table({value: new arrow.Vector([arrow.makeData({type, length: 1, data})])});
}

test.each([
  [9007199254740993n, 0],
  [10000000000000001n, 16],
  [-10000000000000001n, 16]
])('Decimal exact and zero-budget policies reject original coefficient %s at scale %s', (coefficient, scale) => {
  const table = makeDecimalTable(coefficient, scale);
  for (const float64 of [
    {mode: 'exact' as const},
    {mode: 'round' as const, maximumAbsoluteError: 0, maximumRelativeError: 0}
  ]) {
    expect(() => prepareArrowTableForGPUAnalytics(table, {float64})).toThrow(
      'exceeds its precision policy'
    );
  }
});

test('Decimal round policy measures error from the signed scaled coefficient', () => {
  for (const coefficient of [10000000000000001n, -10000000000000001n]) {
    const table = makeDecimalTable(coefficient, 16);
    const prepare = (maximumAbsoluteError: number) =>
      prepareArrowTableForGPUAnalytics(table, {
        float64: {mode: 'round', maximumAbsoluteError, maximumRelativeError: 0}
      });
    expect(() => prepare(0.5e-16)).toThrow('exceeds its precision policy');
    const prepared = prepare(2e-16);
    expect(prepared.table.getChild('value')!.get(0)).toBe(coefficient > 0n ? 1 : -1);
    expect(prepared.conversions[0].maximumAbsoluteError).toBe(1e-16);
    expect(prepared.conversions[0].maximumRelativeError).toBeCloseTo(1e-16, 30);
  }
});

test('wide integer round policy permits one lost unit above the binary64 exact range', () => {
  for (const type of [new arrow.Int64(), new arrow.Uint64()]) {
    const table = new arrow.Table({
      value: arrow.vectorFromArray([9007199254740993n], type)
    });
    const prepared = prepareArrowTableForGPUAnalytics(table, {
      float64: {mode: 'round', maximumAbsoluteError: 1, maximumRelativeError: 0}
    });
    expect(prepared.table.getChild('value')!.get(0)).toBe(9007199254740992);
    expect(prepared.conversions[0].maximumAbsoluteError).toBe(1);
    expect(prepared.conversions[0].maximumRelativeError).toBe(1.1102230246251564e-16);
  }
});

test('Decimal error reporting rounds a rational subnormal once to binary64', () => {
  const prepared = prepareArrowTableForGPUAnalytics(makeDecimalTable(3n, 324), {
    float64: {mode: 'round', maximumAbsoluteError: 1, maximumRelativeError: 0}
  });
  expect(prepared.table.getChild('value')!.get(0)).toBe(0);
  expect(prepared.conversions[0].maximumAbsoluteError).toBe(Number.MIN_VALUE);
  expect(prepared.conversions[0].maximumRelativeError).toBe(1);
});

test.each([2147483647, -2147483648])('Decimal zero remains exact at IPC scale %s', scale => {
  const source = arrow.tableFromIPC(arrow.tableToIPC(makeDecimalTable(0n, scale)));
  expect((source.getChild('value')!.type as arrow.Decimal).scale).toBe(scale);
  const prepared = prepareArrowTableForGPUAnalytics(source, {float64: {mode: 'exact'}});
  expect(prepared.table.getChild('value')!.get(0)).toBe(0);
  expect(prepared.conversions[0].maximumAbsoluteError).toBe(0);
});

test('Decimal extreme underflow applies the policy without constructing the power of ten', () => {
  const source = arrow.tableFromIPC(arrow.tableToIPC(makeDecimalTable(1n, 2147483647)));
  const prepared = prepareArrowTableForGPUAnalytics(source, {
    float64: {mode: 'round', maximumAbsoluteError: 0, maximumRelativeError: 1}
  });
  expect(prepared.table.getChild('value')!.get(0)).toBe(0);
  expect(prepared.conversions[0].maximumAbsoluteError).toBe(0);
  expect(prepared.conversions[0].maximumRelativeError).toBe(1);
});

test('Decimal extreme scales reject by precision policy instead of BigInt allocation', () => {
  for (const scale of [2147483647, -2147483648]) {
    expect(() =>
      prepareArrowTableForGPUAnalytics(makeDecimalTable(1n, scale), {float64: {mode: 'exact'}})
    ).toThrow('exceeds its precision policy');
  }
});

test('Decimal definite underflow preserves sign and enforces budgets at the fast-path boundary', () => {
  for (const scale of [325, 2147483647]) {
    const source = makeDecimalTable(-9n, scale);
    const prepared = prepareArrowTableForGPUAnalytics(source, {
      float64: {mode: 'round', maximumAbsoluteError: Number.MIN_VALUE, maximumRelativeError: 0}
    });
    expect(prepared.table.getChild('value')!.get(0)).toBe(-0);
    expect(prepared.conversions[0].maximumAbsoluteError).toBe(0);
    expect(prepared.conversions[0].maximumRelativeError).toBe(1);
    expect(() =>
      prepareArrowTableForGPUAnalytics(source, {
        float64: {mode: 'round', maximumAbsoluteError: 0, maximumRelativeError: 1 - 2 ** -53}
      })
    ).toThrow('exceeds its precision policy');
  }
});

test.each([
  64, 128, 256
])('Decimal%s preparation supports an explicit exact precision policy', bitWidth => {
  const prepared = prepareArrowTableForGPUAnalytics(makeDecimalTable(-125n, 2, bitWidth), {
    float64: {mode: 'exact'}
  });
  expect(prepared.table.getChild('value')!.get(0)).toBe(-1.25);
});

test('Float64 uses binary64 budget arithmetic while Decimal uses the original rational budget', () => {
  const float64 = {
    mode: 'round' as const,
    maximumAbsoluteError: 0,
    maximumRelativeError: 1 / 16777281
  };
  const source = new arrow.Table({
    value: arrow.vectorFromArray([16777281], new arrow.Float64())
  });
  expect(prepareArrowTableForGPUAnalytics(source, {float64}).table.getChild('value')!.get(0)).toBe(
    16777280
  );
  expect(() => prepareArrowTableForGPUAnalytics(makeDecimalTable(16777281n, 0), {float64})).toThrow(
    'exceeds its precision policy'
  );
});

test('Decimal32 preparation is unsupported even with an explicit policy', () => {
  const type = new arrow.Decimal(0, 9, 32);
  const source = new arrow.Table({
    value: new arrow.Vector([arrow.makeData({type, length: 1, data: new Uint32Array([1])})])
  });
  expect(() => prepareArrowTableForGPUAnalytics(source, {float64: {mode: 'exact'}})).toThrow(
    'Unsupported analytics type'
  );
});

test('Decimal exact preparation preserves negative scales and binary fractions', () => {
  for (const [coefficient, scale, expected] of [
    [125n, 2, 1.25],
    [-125n, 2, -1.25],
    [5n, -1, 50],
    [0n, 16, 0]
  ] as const) {
    const prepared = prepareArrowTableForGPUAnalytics(makeDecimalTable(coefficient, scale), {
      float64: {mode: 'exact'}
    });
    expect(prepared.table.getChild('value')!.get(0)).toBe(expected);
    expect(prepared.conversions[0].maximumAbsoluteError).toBe(0);
  }
});

test('Decimal preparation preserves nulls and batches in the exact path', () => {
  const type = new arrow.Decimal(2, 38, 128);
  const nullable = new arrow.Table({
    value: new arrow.Vector([
      arrow.makeData({
        type,
        length: 2,
        data: new Uint32Array([125, 0, 0, 0, 0, 0, 0, 0]),
        nullBitmap: new Uint8Array([1]),
        nullCount: 1
      })
    ])
  });
  const source = nullable.concat(makeDecimalTable(-125n, 2));
  const prepared = prepareArrowTableForGPUAnalytics(source, {float64: {mode: 'exact'}});
  expect(Array.from(prepared.table.getChild('value')!)).toEqual([1.25, null, -1.25]);
  expect(prepared.table.batches.map(batch => batch.numRows)).toEqual([2, 1]);
});

test('Decimal round policy applies absolute plus relative budgets against the original value', () => {
  // 1.00000001 rounds to 1, so the absolute error is exactly 10^-8.
  const source = makeDecimalTable(100000001n, 8);
  const prepare = (maximumAbsoluteError: number, maximumRelativeError: number) =>
    prepareArrowTableForGPUAnalytics(source, {
      float64: {mode: 'round', maximumAbsoluteError, maximumRelativeError}
    });
  expect(() => prepare(0, 5e-9)).toThrow('exceeds its precision policy');
  expect(prepare(0, 1e-8).table.getChild('value')!.get(0)).toBe(1);
  expect(prepare(5e-9, 5e-9).table.getChild('value')!.get(0)).toBe(1);
  expect(prepare(0, 1e-8).conversions[0].maximumRelativeError).toBeCloseTo(1 / 100000001, 23);
});

test('Decimal narrowing handles subnormals, underflow and overflow without losing diagnostics', () => {
  const prepare = (coefficient: bigint, scale: number) =>
    prepareArrowTableForGPUAnalytics(makeDecimalTable(coefficient, scale), {
      float64: {mode: 'round', maximumAbsoluteError: 1, maximumRelativeError: 0}
    });
  expect(prepare(1n, 45).table.getChild('value')!.get(0)).toBe(2 ** -149);
  expect(prepare(-1n, 46).table.getChild('value')!.get(0)).toBe(-0);
  expect(prepare(1n, 46).conversions[0].maximumAbsoluteError).toBe(1e-46);
  expect(prepare(1n, 46).conversions[0].maximumRelativeError).toBe(1);
  expect(() => prepare(1n, -39)).toThrow('exceeds its precision policy');
});

test('Decimal rounds once to nearest-even on either side of a Float32 midpoint', () => {
  // Midpoint between 1 and 1 + 2^-23 is 1 + 2^-24. Binary64 loses the final decimal digit.
  for (const [coefficient, expected] of [
    [100000005960464477539062499n, 1],
    [100000005960464477539062500n, 1],
    [100000005960464477539062501n, 1 + 2 ** -23]
  ] as const) {
    const prepared = prepareArrowTableForGPUAnalytics(makeDecimalTable(coefficient, 26), {
      float64: {mode: 'round', maximumAbsoluteError: 1e-7, maximumRelativeError: 0}
    });
    expect(prepared.table.getChild('value')!.get(0)).toBe(expected);
  }
});
