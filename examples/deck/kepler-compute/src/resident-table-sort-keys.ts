// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {TableColumnOrderKind} from './resident-table-population-routes';

/**
 * Order-preserving sort keys for table GPU sorting beyond Int32/Uint32.
 *
 * - String dictionaries sort by a rank table: dictionary code to the rank of its value in lexical
 *   (UTF-16 code unit) order, so code order afterwards equals string order. Equal strings share a rank.
 * - 64-bit values (Int64, Uint64, Timestamp, Time64, Date64) and Float64 are split into hi/lo
 *   32-bit words. The words are transformed so unsigned lexicographic (hi, lo) order equals the
 *   value order, then sorted by two stable single-key GPU passes (lo first, then hi).
 */
export type WideSortKind = 'wide-int64' | 'wide-uint64' | 'float64';

export type WideSortWord = {lo: number; hi: number; valid: boolean};

const SIGN_BIT = 0x80000000;

/** Arrow type to the sort family used by the route model. */
export function classifyTableSortType(type: arrow.DataType): TableColumnOrderKind {
  if (arrow.DataType.isDictionary(type)) {
    const kind = classifyTableSortType(type.dictionary);
    return kind === 'dictionary-string' ? kind : 'other-ordered';
  }
  if (arrow.DataType.isFloat(type))
    return type.precision === arrow.Precision.SINGLE
      ? 'float32'
      : type.precision === arrow.Precision.DOUBLE
        ? 'float64'
        : 'other-ordered';
  if (arrow.DataType.isInt(type)) {
    if (type.bitWidth === 32) return type.isSigned ? 'int32' : 'uint32';
    if (type.bitWidth === 64) return type.isSigned ? 'wide-int64' : 'wide-uint64';
    return 'other-ordered';
  }
  if (arrow.DataType.isUtf8(type) || arrow.DataType.isLargeUtf8(type)) return 'dictionary-string';
  if (arrow.DataType.isTimestamp(type)) return 'wide-int64';
  if (arrow.DataType.isTime(type)) return type.bitWidth === 64 ? 'wide-int64' : 'other-ordered';
  if (arrow.DataType.isDate(type))
    return type.unit === arrow.DateUnit.MILLISECOND ? 'wide-int64' : 'other-ordered';
  if (arrow.DataType.isBool(type)) return 'other-ordered';
  return 'unordered';
}

/** One value's order-preserving words; NaN is invalid (sorts last like a missing value). */
export function encodeWideSortWord(kind: WideSortKind, lo: number, hi: number): WideSortWord {
  lo >>>= 0;
  hi >>>= 0;
  if (kind === 'wide-uint64') return {lo, hi, valid: true};
  if (kind === 'wide-int64') return {lo, hi: (hi ^ SIGN_BIT) >>> 0, valid: true};
  const exponentAndSign = hi & 0x7fffffff;
  if (exponentAndSign > 0x7ff00000 || (exponentAndSign === 0x7ff00000 && lo !== 0))
    return {lo: 0, hi: 0, valid: false};
  // Signed zero compares equal.
  if (exponentAndSign === 0 && lo === 0) return {lo: 0, hi: SIGN_BIT, valid: true};
  if (hi & SIGN_BIT) return {lo: ~lo >>> 0, hi: ~hi >>> 0, valid: true};
  return {lo, hi: (hi ^ SIGN_BIT) >>> 0, valid: true};
}

/** Unsigned lexicographic comparison of encoded words. */
export function compareWideSortWords(first: WideSortWord, second: WideSortWord): number {
  return first.hi !== second.hi
    ? first.hi < second.hi
      ? -1
      : 1
    : first.lo < second.lo
      ? -1
      : first.lo > second.lo
        ? 1
        : 0;
}

/**
 * Raw little-endian 32-bit words (lo, hi per row) to separate key arrays. `words` addresses `count`
 * rows starting at `firstRow` (two words per row). Rows flagged missing by `isPresent` become invalid.
 */
export function fillWideSortKeys(props: {
  kind: WideSortKind;
  words: Uint32Array;
  firstRow: number;
  count: number;
  isPresent?: (row: number) => boolean;
  lo: Uint32Array;
  hi: Uint32Array;
  valid: Uint32Array;
  outputOffset: number;
}): void {
  for (let row = 0; row < props.count; row++) {
    const word = encodeWideSortWord(
      props.kind,
      props.words[(props.firstRow + row) * 2],
      props.words[(props.firstRow + row) * 2 + 1]
    );
    const index = props.outputOffset + row;
    props.lo[index] = word.lo;
    props.hi[index] = word.hi;
    props.valid[index] = word.valid && (!props.isPresent || props.isPresent(row)) ? 1 : 0;
  }
}

/**
 * Rank table over a dictionary: ranks[code] is the dense rank of values[code] in UTF-16 code unit
 * order. Sorting rank keys is exactly lexical sorting of the strings.
 */
export function makeDictionaryRankTable(values: readonly string[]): Uint32Array {
  const order = Uint32Array.from({length: values.length}, (_, index) => index);
  order.sort((first, second) =>
    values[first] < values[second] ? -1 : values[first] > values[second] ? 1 : first - second
  );
  const ranks = new Uint32Array(values.length);
  let rank = 0;
  for (let position = 0; position < order.length; position++) {
    if (position > 0 && values[order[position]] !== values[order[position - 1]]) rank++;
    ranks[order[position]] = rank;
  }
  return ranks;
}

/** Whether the prepared dictionary is already in sorted order (rank table is the identity). */
export function isDictionarySorted(values: readonly string[]): boolean {
  for (let index = 1; index < values.length; index++)
    if (values[index] <= values[index - 1]) return false;
  return true;
}

/** Bytes of GPU scratch for one sort program: keys, selection copy, two permutations, gathered pass. */
export function estimateWideSortBytes(rows: number, passes: 1 | 2): number {
  return rows * 4 * (passes === 1 ? 5 : 9) + 65536;
}

/**
 * Publication ceiling for scalar and exact wide sorts. Two compiled stable word passes need
 * about 124 bytes per source row at 300k rows, including keys and compiled scratch. Allow 128
 * bytes per row plus 8 MiB of fixed overhead; measured compilation must still fit this ceiling.
 */
export function getPublishedTableSortMemoryBudgetBytes(rowCount: number): number {
  return 8 * 1024 * 1024 + rowCount * 128;
}
