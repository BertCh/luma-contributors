// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  formatTimestampTicks,
  getTimestampCalendarMilliseconds
} from '@luma-contributors/analysis/dataframe-analysis';

/** Native Kepler field type name (ALL_FIELD_TYPES) for an Arrow type; `other` has no statistics. */
export function getOriginalFieldType(type: arrow.DataType): string {
  if (arrow.DataType.isDictionary(type)) return getOriginalFieldType(type.dictionary);
  if (arrow.DataType.isInt(type)) return 'integer';
  if (arrow.DataType.isFloat(type) || arrow.DataType.isDecimal(type)) return 'real';
  if (arrow.DataType.isUtf8(type) || arrow.DataType.isLargeUtf8(type)) return 'string';
  if (arrow.DataType.isBool(type)) return 'boolean';
  if (arrow.DataType.isTimestamp(type)) return 'timestamp';
  if (arrow.DataType.isDate(type)) return 'date';
  return 'other';
}

/**
 * Plain JavaScript values for the given source rows, as native statistics expect: numbers (64-bit
 * integers narrowed), strings, booleans, timestamps as Unix milliseconds, and null for missing
 * cells. `exact` keeps 64-bit integers as bigint and dates as ISO text (for copying). Only the
 * requested rows are touched. Numeric timestamp presentation floors to the containing millisecond
 * and throws RangeError outside Date's inclusive +/-8,640,000,000,000,000 ms range. Dictionary
 * rows resolve through their own chunk's dictionary before reading the underlying value or ticks.
 */
export function readOriginalColumnValues(
  vector: arrow.Vector,
  rows: ArrayLike<number>,
  exact = false
): unknown[] {
  const fieldType = getOriginalFieldType(vector.type);
  const values = new Array<unknown>(rows.length);
  for (let index = 0; index < rows.length; index++) {
    const original = resolveOriginalColumnRow(vector, rows[index]);
    if (!original) {
      values[index] = null;
      continue;
    }
    const {vector: sourceVector, row: sourceRow} = original;
    if (arrow.DataType.isTimestamp(sourceVector.type)) {
      const ticks = readOriginalTimestampTicks(sourceVector, sourceRow);
      values[index] = exact
        ? formatTimestampTicks(ticks, sourceVector.type.unit)
        : getTimestampCalendarMilliseconds(ticks, sourceVector.type.unit);
      continue;
    }
    const value: unknown = sourceVector.get(sourceRow);
    if (typeof value === 'bigint') values[index] = exact ? value : Number(value);
    else if (value instanceof Date) values[index] = exact ? value.toISOString() : value.getTime();
    else if (fieldType === 'other') values[index] = null;
    else values[index] = value;
  }
  return values;
}

/** Read retained ticks directly; Arrow's Timestamp getter first narrows them to Number. */
export function readOriginalTimestampTicks(vector: arrow.Vector, row: number): bigint {
  const original = resolveOriginalColumnRow(vector, row);
  if (!original) throw new Error('Timestamp requires a valid source row');
  let localRow = original.row;
  for (const chunk of original.vector.data) {
    if (localRow >= chunk.length) {
      localRow -= chunk.length;
      continue;
    }
    const value: unknown = chunk.values[localRow];
    if (typeof value !== 'bigint') throw new Error('Timestamp requires exact 64-bit Arrow storage');
    return value;
  }
  throw new Error('Timestamp source row is outside retained Arrow data');
}

/** Resolve retained dictionary indices without calling the underlying value's lossy Arrow getter. */
export function resolveOriginalColumnRow(
  vector: arrow.Vector,
  row: number
): {vector: arrow.Vector; row: number} | null {
  if (row < 0 || row >= vector.length || !vector.isValid(row)) return null;
  if (!arrow.DataType.isDictionary(vector.type)) return {vector, row};
  let localRow = row;
  for (const chunk of vector.data) {
    if (localRow >= chunk.length) {
      localRow -= chunk.length;
      continue;
    }
    if (!chunk.dictionary) throw new Error('Dictionary source row requires its chunk dictionary');
    return resolveOriginalColumnRow(chunk.dictionary, Number(chunk.values[localRow]));
  }
  return null;
}
