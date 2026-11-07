// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {prepareArrowTableForGPUAnalytics} from '../../graph/src/arrow/gpu/arrow-analytics-normalization';
import {convertWideNumericValue} from '../../graph/src/arrow/gpu/arrow-analytics-wide-numeric';

const POLICY = {mode: 'round' as const, maximumAbsoluteError: 1e-6, maximumRelativeError: 2 ** -23};

test('Int64, Uint64 and Float64 datum columns narrow to Float32 under the policy', () => {
  const table = new arrow.Table({
    signed: arrow.vectorFromArray([BigInt(-90), BigInt(45), null], new arrow.Int64()),
    unsigned: arrow.vectorFromArray([BigInt(0), BigInt(180), BigInt(359)], new arrow.Uint64()),
    double: arrow.vectorFromArray([0.5, 1.25, -3], new arrow.Float64())
  });
  const prepared = prepareArrowTableForGPUAnalytics(table, {float64: POLICY});
  for (const name of ['signed', 'unsigned', 'double'])
    expect(String(prepared.table.schema.fields.find(f => f.name === name)!.type)).toBe('Float32');
  expect(Array.from(prepared.table.getChild('signed')!)).toEqual([-90, 45, null]);
  expect(Array.from(prepared.table.getChild('unsigned')!)).toEqual([0, 180, 359]);
});

test('wide integers need the explicit policy and lossy values stay rejected', () => {
  const table = new arrow.Table({
    big: arrow.vectorFromArray([BigInt('9007199254740993')], new arrow.Int64())
  });
  expect(() => prepareArrowTableForGPUAnalytics(table)).toThrow('explicit precision policy');
  expect(() => prepareArrowTableForGPUAnalytics(table, {float64: {mode: 'exact'}})).toThrow(
    'exceeds its precision policy'
  );
});

test('Decimal values scale from two-complement words', () => {
  const type = new arrow.Decimal(2, 10, 128);
  expect(convertWideNumericValue(type, new Uint32Array([12345, 0, 0, 0]))).toBeCloseTo(123.45);
  expect(
    convertWideNumericValue(
      type,
      new Uint32Array([0xffffffff - 99, 0xffffffff, 0xffffffff, 0xffffffff])
    )
  ).toBeCloseTo(-1);
});
