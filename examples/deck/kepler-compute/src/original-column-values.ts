// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';

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
 * requested rows are touched.
 */
export function readOriginalColumnValues(
  vector: arrow.Vector,
  rows: ArrayLike<number>,
  exact = false
): unknown[] {
  const fieldType = getOriginalFieldType(vector.type);
  const values = new Array<unknown>(rows.length);
  for (let index = 0; index < rows.length; index++) {
    const row = rows[index];
    if (!vector.isValid(row)) {
      values[index] = null;
      continue;
    }
    const value: unknown = vector.get(row);
    if (typeof value === 'bigint') values[index] = exact ? value : Number(value);
    else if (value instanceof Date) values[index] = exact ? value.toISOString() : value.getTime();
    else if (fieldType === 'other') values[index] = null;
    else values[index] = value;
  }
  return values;
}
