// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';

import {clamp} from '@math.gl/core';

export const MAXIMUM_ORIGINAL_TABLE_CELLS = 16_384;
export const MAXIMUM_ORIGINAL_TABLE_CHARACTERS = 1_048_576;
const MAXIMUM_CELL_CHARACTERS = 4096;
const PREVIEW_SUFFIX = '… [preview]';

export function formatOriginalTableValue(vector: arrow.Vector, row: number): string {
  return formatOriginalTableCell(vector, row).text;
}

/**
 * Tooltip value: finite numbers stay numeric so native field formats and compare-mode deltas apply;
 * everything else keeps the bounded display string.
 */
export function readOriginalTooltipValue(vector: arrow.Vector, row: number): string | number {
  if (
    (arrow.DataType.isFloat(vector.type) || arrow.DataType.isInt(vector.type)) &&
    Number.isSafeInteger(row) &&
    row >= 0 &&
    row < vector.length &&
    vector.isValid(row)
  ) {
    const value: unknown = vector.get(row);
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return formatOriginalTableValue(vector, row);
}

/** Bound display work without materializing an entire nested Arrow vector through toJSON/toArray. */
export function formatOriginalTableCell(
  vector: arrow.Vector,
  row: number,
  remainingCharacters = MAXIMUM_CELL_CHARACTERS
): {text: string; truncated: boolean} {
  if (!Number.isSafeInteger(row) || row < 0 || row >= vector.length)
    throw new Error('Source row is outside retained Arrow data');
  if (!vector.isValid(row)) return {text: 'null', truncated: false};
  const maximum = clamp(remainingCharacters, 0, MAXIMUM_CELL_CHARACTERS);
  if (maximum < PREVIEW_SUFFIX.length) return {text: '', truncated: true};
  const limit = maximum - PREVIEW_SUFFIX.length;
  let text = '';
  let truncated = false;
  let nodes = 0;
  const append = (value: string) => {
    const available = limit - text.length;
    text += value.slice(0, available);
    if (value.length > available) truncated = true;
  };
  const quote = (value: string) => {
    const available = limit - text.length;
    append(JSON.stringify(value.slice(0, available)));
    if (value.length > available) truncated = true;
  };
  const visit = (value: unknown, depth: number) => {
    if (truncated) return;
    if (++nodes > 2048 || depth > 12) {
      truncated = true;
      return;
    }
    if (value === null || value === undefined) {
      append('null');
      return;
    }
    if (typeof value === 'string' || typeof value === 'bigint') {
      quote(String(value));
      return;
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
      append(String(value));
      return;
    }
    if (value instanceof Date) {
      quote(value.toISOString());
      return;
    }
    if (value instanceof arrow.Vector || Array.isArray(value) || value instanceof Uint8Array) {
      append('[');
      for (let index = 0; index < value.length && !truncated; index++) {
        if (index) append(',');
        visit(value instanceof arrow.Vector ? value.get(index) : value[index], depth + 1);
      }
      if (!truncated) append(']');
      return;
    }
    append('{');
    let count = 0;
    if (value instanceof arrow.StructRow || value instanceof arrow.MapRow) {
      for (const [key, item] of value) {
        if (truncated) break;
        if (count++) append(',');
        quote(String(key));
        append(':');
        visit(item, depth + 1);
      }
    } else if (typeof value === 'object') {
      for (const key in value) {
        if (truncated) break;
        if (!Object.hasOwn(value, key)) continue;
        if (count++) append(',');
        quote(key);
        append(':');
        visit(Reflect.get(value, key), depth + 1);
      }
    }
    if (!truncated) append('}');
  };
  let localRow = row;
  for (const data of vector.data) {
    if (localRow >= data.length) {
      localRow -= data.length;
      continue;
    }
    if (arrow.DataType.isDictionary(vector.type)) {
      const dictionary = data.dictionary;
      if (!dictionary) throw new Error('Original dictionary values are unavailable');
      return formatOriginalTableCell(
        dictionary,
        Number(data.values[localRow]),
        remainingCharacters
      );
    }
    if (arrow.DataType.isTimestamp(vector.type)) {
      const value = data.values[localRow];
      if (typeof value !== 'bigint')
        throw new Error('Original timestamp requires exact 64-bit Arrow storage');
      append(
        `${value} ${['s', 'ms', 'us', 'ns'][vector.type.unit]} since Unix epoch${vector.type.timezone ? ` (${vector.type.timezone})` : ' (timezone unspecified)'}`
      );
    } else if (arrow.DataType.isUtf8(vector.type) || arrow.DataType.isLargeUtf8(vector.type)) {
      const start = Number(data.valueOffsets[localRow]);
      const end = Number(data.valueOffsets[localRow + 1]);
      const boundedEnd = Math.min(end, start + maximum * 4);
      append(
        new TextDecoder().decode(data.values.subarray(start, boundedEnd), {
          stream: boundedEnd < end
        })
      );
      if (boundedEnd < end) truncated = true;
    } else {
      const value: unknown = vector.get(row);
      if (value === null || value === undefined) append('null');
      else if (
        typeof value === 'string' ||
        typeof value === 'number' ||
        typeof value === 'bigint' ||
        typeof value === 'boolean'
      )
        append(String(value));
      else if (value instanceof Date) append(value.toISOString());
      else visit(value, 0);
    }
    return {text: text + (truncated ? PREVIEW_SUFFIX : ''), truncated};
  }
  throw new Error('Source row is outside retained Arrow data');
}
