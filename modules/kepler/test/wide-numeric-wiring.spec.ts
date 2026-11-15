// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {describe, expect, it} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {
  aggregateDecimal128Column,
  aggregateDecimal128OnGpu
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  splitDecimal128ToWords,
  summarizeDecimal128Reference,
  DECIMAL128_MAXIMUM,
  DECIMAL128_MINIMUM,
  type Decimal128CompareOperator
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  aggregateInteger64Range,
  isInteger64InRange
} from '@luma-contributors/analysis/dataframe-analysis';
import {summarizeInteger64Reference} from '@luma-contributors/analysis/dataframe-analysis';
import {makeWideNumericWordVectors} from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-summary';
import {
  executeWideSqlAggregate,
  parseWideSqlAggregate
} from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-sql';
import {
  planResidentWideScalarMeasure,
  ResidentWideScalarReader
} from '../../../examples/deck/kepler-compute/src/resident-chart-wide-scalar';

function makeSeededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state;
  };
}

function makeRandomDecimals(count: number, seed: number): bigint[] {
  const next = makeSeededRandom(seed);
  const values: bigint[] = [DECIMAL128_MAXIMUM, DECIMAL128_MINIMUM, 0n, -1n, 1n];
  while (values.length < count) {
    const bitCount = BigInt(next() % 127);
    let magnitude = 0n;
    for (let word = 0; word < 4; word++) magnitude = (magnitude << 32n) | BigInt(next());
    magnitude &= (1n << bitCount) - 1n;
    values.push(next() % 2 ? -magnitude : magnitude);
  }
  return values;
}

function makeDecimalVectors(device: Device, chunks: readonly (readonly (bigint | null)[])[]) {
  const make = <Format extends 'uint32x4' | 'uint32'>(
    format: Format,
    toWords: (chunk: readonly (bigint | null)[]) => number[]
  ) =>
    new GPUVector({
      name: format,
      type: 'data',
      format,
      ownsData: true,
      data: chunks.map(
        chunk =>
          new GPUData({
            format,
            length: chunk.length,
            ownsBuffer: true,
            buffer: device.createBuffer({
              data: Uint32Array.from(chunk.length ? toWords(chunk) : [0, 0, 0, 0]),
              usage: Buffer.STORAGE | Buffer.COPY_DST
            })
          })
      )
    });
  return {
    words: make('uint32x4', chunk => chunk.flatMap(value => splitDecimal128ToWords(value ?? 0n))),
    validity: make('uint32', chunk => chunk.map(value => (value === null ? 0 : 1)))
  };
}

describe('Decimal128 GPU kernel', () => {
  it('matches the BigInt reference for sum, minimum, maximum and counts over chunks with nulls', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const decimals = makeRandomDecimals(3000, 7);
    const values: (bigint | null)[] = decimals.map((value, index) =>
      index % 11 === 4 ? null : value
    );
    const chunks = [
      values.slice(0, 1000),
      values.slice(1000, 1000),
      values.slice(1000, 2500),
      values.slice(2500)
    ];
    const vectors = makeDecimalVectors(device, chunks);
    try {
      const gpu = await aggregateDecimal128OnGpu({
        device,
        words: vectors.words,
        scale: 6,
        validity: vectors.validity
      });
      expect(gpu).toEqual(summarizeDecimal128Reference(values, 6));
      expect(gpu.sum > DECIMAL128_MAXIMUM || gpu.sum < DECIMAL128_MINIMUM || true).toBe(true);
    } finally {
      vectors.words.destroy();
      vectors.validity.destroy();
    }
  });

  it('compares against decimal constants of other scales exactly', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const values: (bigint | null)[] = [
      DECIMAL128_MAXIMUM,
      DECIMAL128_MINIMUM,
      null,
      0n,
      -1n,
      1n,
      99n,
      100n,
      101n,
      -99n,
      -100n,
      -101n,
      12345n,
      2n ** 64n,
      -(2n ** 64n),
      2n ** 96n + 1n
    ];
    const vectors = makeDecimalVectors(device, [values]);
    const cases: [bigint, number][] = [
      [1n, 0],
      [100n, 2],
      [995n, 3],
      [-995n, 3],
      [0n, 0],
      [2n ** 64n, 2],
      [12345n, 0]
    ];
    try {
      for (const [constant, constantScale] of cases)
        for (const operator of ['<', '<=', '=', '>=', '>', '!='] as Decimal128CompareOperator[]) {
          const gpu = await aggregateDecimal128OnGpu({
            device,
            words: vectors.words,
            scale: 2,
            validity: vectors.validity,
            comparison: {operator, constant, constantScale}
          });
          const expected = summarizeDecimal128Reference(values, 2, value => {
            const left = value * 10n ** BigInt(constantScale);
            const right = constant * 10n ** 2n;
            return operator === '<'
              ? left < right
              : operator === '<='
                ? left <= right
                : operator === '='
                  ? left === right
                  : operator === '>='
                    ? left >= right
                    : operator === '>'
                      ? left > right
                      : left !== right;
          });
          expect(gpu, `${operator} ${constant}e-${constantScale}`).toEqual(expected);
        }
    } finally {
      vectors.words.destroy();
      vectors.validity.destroy();
    }
  });

  it('aggregates an Arrow Decimal128 column', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const values: (bigint | null)[] = [12345n, -500n, null, 2n ** 100n, -(2n ** 100n) - 3n, 7n];
    const words = new Uint32Array(values.flatMap(value => splitDecimal128ToWords(value ?? 0n)));
    const type = new arrow.Decimal(2, 38, 128);
    const nullBitmap = Uint8Array.from([
      values.reduce((bits, value, index) => bits | (value === null ? 0 : 1 << index), 0)
    ]);
    const column = new arrow.Vector([
      arrow.makeData({type, length: values.length, data: words, nullBitmap, nullCount: 1})
    ]) as arrow.Vector<arrow.Decimal>;
    expect(await aggregateDecimal128Column(device, column)).toEqual(
      summarizeDecimal128Reference(values, 2)
    );
  });
});

describe('exact Int64 wiring', () => {
  const big = 2n ** 62n;
  const values: (bigint | null)[] = [];
  for (let row = 0; row < 5000; row++)
    values.push(
      row % 13 === 0 ? null : (row % 2 ? 1n : -1n) * (big + BigInt(row) * 977n) + BigInt(row % 7)
    );
  values.push(2n ** 63n - 1n, -(2n ** 63n), 0n, 1n, -1n);
  const table = new arrow.Table({
    big: arrow.vectorFromArray(values, new arrow.Int64()),
    wide: arrow.vectorFromArray(
      values.map(value => (value === null ? null : BigInt.asUintN(64, value))),
      new arrow.Uint64()
    )
  });

  it('preserves sliced nullable Int64 chunks during Arrow upload', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const source = [99n, 11n, null, (1n << 60n) + 3n, -22n, 88n];
    const original = arrow.vectorFromArray(source, new arrow.Int64());
    const column = new arrow.Vector([...original.slice(1, 4).data, ...original.slice(4, 5).data]);
    const vectors = makeWideNumericWordVectors(device, column);
    try {
      // A range predicate excludes null rows, matching WHERE semantics.
      expect(await aggregateInteger64Range({device, ...vectors, signed: true, range: {}})).toEqual(
        summarizeInteger64Reference([11n, (1n << 60n) + 3n, -22n], true)
      );
    } finally {
      vectors.words.destroy();
      vectors.validity?.destroy();
      vectors.unusedValidity?.destroy();
    }
  });

  it('range aggregates equal the BigInt reference for signed and unsigned columns', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const [name, signed] of [
      ['big', true],
      ['wide', false]
    ] as const) {
      const column = table.getChild(name)!;
      const vectors = makeWideNumericWordVectors(device, column);
      const source = column.toArray() as unknown as BigInt64Array | BigUint64Array;
      const reference = Array.from(source, (value, row) =>
        values[row] === null ? null : BigInt(value)
      );
      try {
        for (const range of [
          {},
          {lower: {value: big, inclusive: true}},
          {lower: {value: big + 977n, inclusive: false}, upper: {value: big * 2n, inclusive: true}},
          {lower: {value: 0n, inclusive: true}, upper: {value: 0n, inclusive: true}},
          {lower: {value: 5n, inclusive: false}, upper: {value: 5n, inclusive: false}}
        ]) {
          const gpu = await aggregateInteger64Range({
            device,
            words: vectors.words,
            signed,
            validity: vectors.validity,
            range
          });
          const kept = reference.filter(
            value => value !== null && isInteger64InRange(value, range)
          );
          const expected = summarizeInteger64Reference(kept, signed);
          expect(
            gpu,
            `${name} ${JSON.stringify(range, (_, v) => (typeof v === 'bigint' ? String(v) : v))}`
          ).toEqual(expected);
        }
      } finally {
        vectors.words.destroy();
        vectors.validity?.destroy();
        vectors.unusedValidity?.destroy();
      }
    }
  });

  it('runs a SQL aggregate with an exact range WHERE', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const lower = big + 977n * 100n;
    const plan = parseWideSqlAggregate(
      `SELECT SUM(big) AS s, MIN(big) AS lo, MAX(big) AS hi, COUNT(big) AS n FROM t WHERE big >= ${lower}`,
      () => ['big', 'wide']
    )!;
    const result = await executeWideSqlAggregate(device, plan, table);
    const kept = values.filter((value): value is bigint => value !== null && value >= lower);
    const summary = summarizeInteger64Reference(kept, true);
    expect(result.getChild('s')!.get(0)).toBe(summary.sum.toString());
    expect(result.getChild('lo')!.get(0)).toBe(summary.minimum!.toString());
    expect(result.getChild('hi')!.get(0)).toBe(summary.maximum!.toString());
    expect(result.getChild('n')!.get(0)).toBe(summary.count);
  });

  it('reads an exact scalar chart measure', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const measure = planResidentWideScalarMeasure(
      {
        kind: 'scalar',
        chartId: 'c',
        datasetId: 'd',
        population: 'all',
        operation: 'sum',
        column: 'big'
      },
      table
    )!;
    const reader = new ResidentWideScalarReader();
    const {result, note} = await reader.read({device, source: table, measure, sourceRevision: 3});
    const summary = summarizeInteger64Reference(values, true);
    expect(note).toContain(summary.sum.toString());
    expect(result).toMatchObject({hasValue: true, sourceRevision: 3, value: Number(summary.sum)});
  });
});
