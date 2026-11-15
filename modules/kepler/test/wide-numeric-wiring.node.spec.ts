// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {
  combineDecimal128SumAccumulators,
  compareDecimal128Reference,
  DECIMAL128_MAXIMUM,
  DECIMAL128_MINIMUM,
  formatDecimal128,
  joinWordsToDecimal128,
  normalizeDecimal128Comparison,
  splitDecimal128ToWords,
  summarizeDecimal128Reference,
  type Decimal128CompareOperator
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  isInteger64InRange,
  makeInteger64RangeSelectionExpression,
  resolveInteger64Range
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  makeWideSqlAggregateTable,
  parseWideSqlAggregate
} from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-sql';
import {summarizeInteger64Reference} from '@luma-contributors/analysis/dataframe-analysis';
import {planResidentWideScalarMeasure} from '../../../examples/deck/kepler-compute/src/resident-chart-wide-scalar';
import {makeDecimal128SelectionExpression} from '@luma-contributors/analysis/dataframe-analysis';

const OPERATORS: Decimal128CompareOperator[] = ['<', '<=', '=', '>=', '>', '!='];

/** Independent oracle: cross-multiplied rational comparison, no normalization. */
function compareRational(
  operator: Decimal128CompareOperator,
  value: bigint,
  scale: number,
  constant: bigint,
  constantScale: number
): boolean {
  const left = value * 10n ** BigInt(constantScale);
  const right = constant * 10n ** BigInt(scale);
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
}

describe('Decimal128 reference', () => {
  it('round-trips words including the extremes', () => {
    for (const value of [
      0n,
      1n,
      -1n,
      DECIMAL128_MAXIMUM,
      DECIMAL128_MINIMUM,
      2n ** 64n,
      -(2n ** 96n) + 7n
    ])
      expect(joinWordsToDecimal128(splitDecimal128ToWords(value))).toBe(value);
    expect(splitDecimal128ToWords(-1n)).toEqual([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
  });

  it('formats scaled decimals', () => {
    expect(formatDecimal128(-12345n, 2)).toBe('-123.45');
    expect(formatDecimal128(5n, 3)).toBe('0.005');
    expect(formatDecimal128(42n, 0)).toBe('42');
  });

  it('combines word-wise sum accumulators exactly', () => {
    const values = [DECIMAL128_MAXIMUM, DECIMAL128_MINIMUM, -5n, 2n ** 100n, 123456789n];
    const sums = [0, 0, 0, 0];
    const carries = [0, 0, 0, 0];
    let negatives = 0;
    for (const value of values) {
      const words = splitDecimal128ToWords(value);
      words.forEach((word, index) => {
        const old = sums[index];
        sums[index] = (old + word) >>> 0;
        if (sums[index] < old) carries[index]++;
      });
      if (value < 0n) negatives++;
    }
    expect(combineDecimal128SumAccumulators(sums, carries, negatives)).toBe(
      values.reduce((total, value) => total + value, 0n)
    );
  });

  it('normalizes comparisons exactly against a rational oracle', () => {
    const values = [-1000n, -101n, -100n, -99n, -1n, 0n, 1n, 99n, 100n, 101n, 1000n];
    const constants: [bigint, number][] = [
      [1n, 0],
      [-1n, 0],
      [995n, 3],
      [-995n, 3],
      [10n, 1],
      [5n, 2],
      [123n, 5],
      [0n, 4]
    ];
    for (const scale of [0, 2, 4])
      for (const [constant, constantScale] of constants)
        for (const operator of OPERATORS) {
          const expected = values.filter(value =>
            compareRational(operator, value, scale, constant, constantScale)
          ).length;
          expect(compareDecimal128Reference(values, scale, operator, constant, constantScale)).toBe(
            expected
          );
        }
  });

  it('turns out-of-range thresholds into constants', () => {
    const huge = DECIMAL128_MAXIMUM + 1n;
    expect(normalizeDecimal128Comparison('<', huge, 0, 0)).toEqual({
      kind: 'constant',
      result: true
    });
    expect(normalizeDecimal128Comparison('>', huge, 0, 0)).toEqual({
      kind: 'constant',
      result: false
    });
    expect(normalizeDecimal128Comparison('=', -huge - 1n, 0, 0)).toEqual({
      kind: 'constant',
      result: false
    });
    expect(normalizeDecimal128Comparison('!=', -huge - 1n, 0, 0)).toEqual({
      kind: 'constant',
      result: true
    });
    expect(normalizeDecimal128Comparison('=', 15n, 1, 0)).toEqual({
      kind: 'constant',
      result: false
    });
    expect(
      makeDecimal128SelectionExpression({
        operator: '>',
        constant: 3n,
        constantScale: 0,
        columnScale: 2
      })
    ).toContain('decimalLess');
  });

  it('summarizes with nulls and a filter', () => {
    const summary = summarizeDecimal128Reference([5n, null, -7n, 20n], 2, value => value > -10n);
    expect(summary).toMatchObject({count: 3, nullCount: 0, minimum: -7n, maximum: 20n, sum: 18n});
  });
});

describe('Int64 range filter', () => {
  it('resolves exclusive bounds and empty ranges', () => {
    expect(
      resolveInteger64Range(
        {lower: {value: 5n, inclusive: false}, upper: {value: 9n, inclusive: false}},
        true
      )
    ).toEqual({minimum: 6n, maximum: 8n});
    expect(
      resolveInteger64Range(
        {lower: {value: 9n, inclusive: true}, upper: {value: 3n, inclusive: true}},
        true
      )
    ).toBeNull();
    expect(resolveInteger64Range({upper: {value: -1n, inclusive: true}}, false)).toBeNull();
  });

  it('emits exact word comparisons', () => {
    const expression = makeInteger64RangeSelectionExpression(
      {lower: {value: 2n ** 53n + 1n, inclusive: true}},
      true
    );
    expect(expression).toContain('word64LessSigned');
    expect(expression).toContain('vec2<u32>(1u, 2097152u)');
    expect(makeInteger64RangeSelectionExpression({}, false)).toBe('true');
    expect(
      makeInteger64RangeSelectionExpression(
        {lower: {value: 5n, inclusive: true}, upper: {value: 4n, inclusive: true}},
        true
      )
    ).toBe('false');
    expect(isInteger64InRange(5n, {lower: {value: 5n, inclusive: false}})).toBe(false);
    expect(isInteger64InRange(5n, {lower: {value: 5n, inclusive: true}})).toBe(true);
  });
});

describe('wide SQL aggregates', () => {
  const columns = (name: string) => (name === 'trips' ? ['Big', 'other'] : undefined);

  it('parses supported statements', () => {
    const plan = parseWideSqlAggregate(
      'SELECT SUM(big) AS total, MIN("Big"), count(*) FROM trips WHERE big >= 10 AND big < 100;',
      columns
    );
    expect(plan?.items.map(item => [item.functionName, item.column, item.outputName])).toEqual([
      ['sum', 'Big', 'total'],
      ['min', 'Big', 'min(Big)'],
      ['count', null, 'count_star()']
    ]);
    expect(plan?.filter).toEqual({
      column: 'Big',
      range: {lower: {value: 10n, inclusive: true}, upper: {value: 100n, inclusive: false}}
    });
  });

  it('declines everything else', () => {
    for (const sql of [
      'SELECT SUM(big) FROM other_table',
      'SELECT AVG(big) FROM trips',
      'SELECT big FROM trips',
      'SELECT SUM(big) FROM trips GROUP BY other',
      'SELECT SUM(big) FROM trips WHERE other > 3',
      'SELECT SUM(big), SUM(other) FROM trips WHERE big > 3',
      'SELECT COUNT(*) FROM trips',
      'SELECT SUM(big) FROM trips WHERE big > 1 OR big < 0'
    ])
      expect(parseWideSqlAggregate(sql, columns), sql).toBeNull();
  });

  it('tightens ranges and builds the exact result table', () => {
    const plan = parseWideSqlAggregate(
      'SELECT SUM(big), MAX(big), COUNT(big) FROM trips WHERE big > 1 AND big > 5 AND big <= 50 AND big <= 40',
      columns
    )!;
    expect(plan.filter!.range).toEqual({
      lower: {value: 5n, inclusive: false},
      upper: {value: 40n, inclusive: true}
    });
    const summary = summarizeInteger64Reference([2n ** 62n, 2n ** 62n, 3n], true);
    const table = makeWideSqlAggregateTable(plan, new Map([['Big', summary]]));
    expect(table.getChildAt(0)!.get(0)).toBe((2n ** 63n + 3n).toString());
    expect(table.getChildAt(1)!.get(0)).toBe((2n ** 62n).toString());
    expect(table.getChildAt(2)!.get(0)).toBe(3);
    const empty = makeWideSqlAggregateTable(
      plan,
      new Map([['Big', summarizeInteger64Reference([], true)]])
    );
    expect(empty.getChildAt(0)!.get(0)).toBeNull();
    expect(empty.getChildAt(2)!.get(0)).toBe(0);
  });
});

describe('wide scalar chart measure', () => {
  const source = new arrow.Table({
    big: arrow.makeVector(new BigInt64Array([1n])),
    small: arrow.makeVector(new Int32Array([1]))
  });
  const scalar = (column: string, operation: string, population = 'all') =>
    ({kind: 'scalar', chartId: 'c', datasetId: 'd', population, operation, column}) as never;

  it('applies only to whole-population sum, mean, min and max on 64-bit integers', () => {
    expect(planResidentWideScalarMeasure(scalar('big', 'sum'), source)).toEqual({
      column: 'big',
      signed: true,
      operation: 'sum'
    });
    expect(planResidentWideScalarMeasure(scalar('big', 'mean'), source)?.operation).toBe('mean');
    expect(planResidentWideScalarMeasure(scalar('big', 'median'), source)).toBeNull();
    expect(planResidentWideScalarMeasure(scalar('big', 'sum', 'selected'), source)).toBeNull();
    expect(planResidentWideScalarMeasure(scalar('small', 'sum'), source)).toBeNull();
  });
});
