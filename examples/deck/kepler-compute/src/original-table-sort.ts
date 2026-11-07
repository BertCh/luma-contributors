// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';

/** Rich original sorting reads every sorted row's key; refuse populations above this bound. */
export const MAXIMUM_ORIGINAL_SORT_ROWS = 262_144;

export type OriginalSortKey = number | bigint | string;
export type OriginalSortDirection = 'ascending' | 'descending';

/** Types whose complete values have a total order: scalars, strings, dates, timestamps, 64-bit. */
export function isOriginalSortType(type: arrow.DataType): boolean {
  if (arrow.DataType.isDictionary(type)) return isOriginalSortType(type.dictionary);
  return (
    arrow.DataType.isInt(type) ||
    arrow.DataType.isFloat(type) ||
    arrow.DataType.isBool(type) ||
    arrow.DataType.isUtf8(type) ||
    arrow.DataType.isLargeUtf8(type) ||
    arrow.DataType.isDate(type) ||
    arrow.DataType.isTimestamp(type) ||
    arrow.DataType.isTime(type)
  );
}

/**
 * Exact key of one non-null cell. Timestamps and 64-bit integers stay bigint (never rounded to
 * double milliseconds), strings compare by UTF-16 code units, dictionaries compare by value.
 * Returns undefined for a missing or NaN cell, which sorts last in both directions.
 */
export function readOriginalSortKey(
  vector: arrow.Vector,
  row: number
): OriginalSortKey | undefined {
  if (!vector.isValid(row)) return undefined;
  let local = row;
  for (const data of vector.data) {
    if (local >= data.length) {
      local -= data.length;
      continue;
    }
    if (arrow.DataType.isDictionary(vector.type)) {
      const dictionary = data.dictionary;
      return dictionary ? readOriginalSortKey(dictionary, Number(data.values[local])) : undefined;
    }
    if (arrow.DataType.isTimestamp(vector.type) || arrow.DataType.isTime(vector.type)) {
      const value = data.values[local];
      return typeof value === 'bigint' ? value : Number(value);
    }
    break;
  }
  const value: unknown = vector.get(row);
  if (typeof value === 'bigint' || typeof value === 'string') return value;
  if (typeof value === 'number') return Number.isNaN(value) ? undefined : value;
  if (typeof value === 'boolean') return value ? 1 : 0;
  if (value instanceof Date) return value.getTime();
  return undefined;
}

/** Mixed bigint and number compare by value; keys within one column share a representation. */
export function compareOriginalSortKeys(first: OriginalSortKey, second: OriginalSortKey): number {
  return first < second ? -1 : first > second ? 1 : 0;
}

/**
 * Stable exact order of `rows` by one original column: missing values last for both directions,
 * ties keep the incoming row order. Keys are read once, so work is O(n log n) comparisons.
 */
export function sortOriginalRows(
  vector: arrow.Vector,
  rows: ArrayLike<number>,
  direction: OriginalSortDirection
): Uint32Array {
  if (rows.length > MAXIMUM_ORIGINAL_SORT_ROWS)
    throw new Error('Table operation exceeds its 262,144-row bound'); // narrow the selection
  const keys = new Array<OriginalSortKey | undefined>(rows.length);
  const order = new Uint32Array(rows.length);
  for (let index = 0; index < rows.length; index++) {
    keys[index] = readOriginalSortKey(vector, rows[index]);
    order[index] = index;
  }
  const sign = direction === 'ascending' ? 1 : -1;
  order.sort((first, second) => {
    const left = keys[first];
    const right = keys[second];
    if (left === undefined || right === undefined) {
      if (left === right) return first - second;
      return left === undefined ? 1 : -1;
    }
    return sign * compareOriginalSortKeys(left, right) || first - second;
  });
  const result = new Uint32Array(rows.length);
  for (let index = 0; index < order.length; index++) result[index] = rows[order[index]];
  return result;
}
