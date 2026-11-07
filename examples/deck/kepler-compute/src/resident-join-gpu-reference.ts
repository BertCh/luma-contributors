// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  RESIDENT_JOIN_NO_KEY,
  RESIDENT_JOIN_NO_ROW,
  type ResidentJoinColumnFormat,
  type ResidentJoinType
} from './resident-join-plan';

/**
 * Step-for-step CPU mirror of the GPU Join pipeline in resident-join-gpu-kernels.ts. Each function
 * corresponds to one kernel or graph primitive so node tests can check the algorithm against the
 * pinned native Join and the dictionary oracle without a GPU. Status bits match the summary word.
 */
export const RESIDENT_JOIN_STATUS_FLOAT_KEY = 1;
export const RESIDENT_JOIN_STATUS_DICTIONARY_CODE = 2;
export const RESIDENT_JOIN_STATUS_OUTPUT_OVERFLOW = 4;

export type ResidentJoinKeyMode = 'string' | ResidentJoinColumnFormat;

/** Mirrors the key kernel: canonical uint32 key and a participates-in-matching flag per row. */
export function canonicalizeResidentJoinKeys(props: {
  words: Uint32Array;
  validity?: Uint32Array;
  mode: ResidentJoinKeyMode;
  /** String mode: dictionary code to shared key. */
  table?: Uint32Array;
  dropHighKeys?: boolean;
}): {key: Uint32Array; ok: Uint32Array; status: number} {
  const {words, validity, mode, table} = props;
  const key = new Uint32Array(words.length);
  const ok = new Uint32Array(words.length);
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  let status = 0;
  for (let row = 0; row < words.length; row++) {
    if (validity && !validity[row]) continue;
    const word = words[row];
    if (mode === 'string') {
      if (word >= table!.length) status |= RESIDENT_JOIN_STATUS_DICTIONARY_CODE;
      else {
        key[row] = table![word];
        ok[row] = key[row] !== RESIDENT_JOIN_NO_KEY ? 1 : 0;
      }
    } else if (mode === 'float32') {
      const value = floats[row];
      if (Math.abs(value) <= 16_777_216 && value === Math.floor(value)) {
        key[row] = value | 0;
        ok[row] = 1;
      } else status |= RESIDENT_JOIN_STATUS_FLOAT_KEY;
    } else {
      key[row] = word;
      ok[row] = props.dropHighKeys && mode === 'uint32' && word >= 0x80000000 ? 0 : 1;
    }
  }
  return {key, ok, status};
}

/** Mirrors the stable sorts and run flags: valid right rows in key order, then the rest. */
export function sortResidentJoinRightRows(rightKey: Uint32Array, rightOk: Uint32Array) {
  const rowCount = rightKey.length;
  // Sort 1: stable by key. Sort 2: stable by not-ok flag, which keeps key order among valid rows.
  const byKey = Array.from({length: rowCount}, (_, row) => row).sort(
    (first, second) => rightKey[first] - rightKey[second] || first - second
  );
  const sortedRows = Uint32Array.from(
    [...byKey].sort((first, second) => (rightOk[first] ? 0 : 1) - (rightOk[second] ? 0 : 1))
  );
  const sortedKeys = new Uint32Array(rowCount);
  const okIncluded = new Uint32Array(rowCount);
  const runStart = new Uint32Array(rowCount);
  for (let position = 0; position < rowCount; position++) {
    const row = sortedRows[position];
    sortedKeys[position] = rightKey[row];
    okIncluded[position] = rightOk[row] ? 1 : 0;
    // Valid rows are a prefix, so the previous position is valid whenever this one is.
    runStart[position] =
      rightOk[row] && (position === 0 || rightKey[sortedRows[position - 1]] !== rightKey[row])
        ? 1
        : 0;
  }
  return {
    sortedRows,
    sortedKeys,
    validCount: okIncluded.reduce((total, value) => total + value, 0),
    runIndex: Uint32Array.from(inclusiveScan(runStart))
  };
}

/** Mirrors the probe kernel, scans, summary kernel and fill kernel. */
export function planResidentJoinPairsOnGpu(props: {
  joinType: ResidentJoinType;
  leftKey: Uint32Array;
  leftOk: Uint32Array;
  rightKey: Uint32Array;
  rightOk: Uint32Array;
  duplicateRightMatches?: 'all' | 'first' | 'last';
}): {leftRows: Uint32Array; rightRows: Uint32Array; rowCount: number; status: number} {
  const {joinType, leftKey, leftOk, rightKey, rightOk} = props;
  const leftCount = leftKey.length;
  const rightCount = rightKey.length;
  const {sortedRows, sortedKeys, validCount, runIndex} = sortResidentJoinRightRows(
    rightKey,
    rightOk
  );
  const lowerBound = new Uint32Array(leftCount);
  const matches = new Uint32Array(leftCount);
  const emit = new Uint32Array(leftCount);
  const matchedRun = new Uint32Array(rightCount);
  for (let row = 0; row < leftCount; row++) {
    if (leftOk[row] && validCount) {
      const key = leftKey[row];
      let low = 0;
      let high = validCount;
      while (low < high) {
        const middle = low + ((high - low) >>> 1);
        if (sortedKeys[middle] < key) low = middle + 1;
        else high = middle;
      }
      lowerBound[row] = low;
      let upLow = low;
      let upHigh = validCount;
      while (upLow < upHigh) {
        const middle = upLow + ((upHigh - upLow) >>> 1);
        if (sortedKeys[middle] <= key) upLow = middle + 1;
        else upHigh = middle;
      }
      matches[row] = upLow - low;
      if (matches[row] && props.duplicateRightMatches === 'first') matches[row] = 1;
      else if (matches[row] && props.duplicateRightMatches === 'last') {
        lowerBound[row] = upLow - 1;
        matches[row] = 1;
      }
      if (matches[row]) matchedRun[runIndex[low] - 1] = 1;
    }
    emit[row] = matches[row] || (joinType === 'INNER' ? 0 : 1);
  }
  const unmatched = new Uint32Array(rightCount);
  for (let position = 0; position < rightCount; position++) {
    const matched = position < validCount ? matchedRun[runIndex[position] - 1] : 0;
    unmatched[sortedRows[position]] = joinType === 'FULL' && !matched ? 1 : 0;
  }
  const leftInclusive = inclusiveScan(emit);
  const unmatchedInclusive = inclusiveScan(unmatched);
  const leftTotal = leftInclusive[leftCount - 1];
  const unmatchedTotal = unmatchedInclusive[rightCount - 1];
  const rowCount = leftTotal + unmatchedTotal;
  const status = leftTotal >= 0x100000000 ? RESIDENT_JOIN_STATUS_OUTPUT_OVERFLOW : 0;
  const leftRows = new Uint32Array(rowCount);
  const rightRows = new Uint32Array(rowCount);
  for (let index = 0; index < rowCount; index++) {
    let leftRow = RESIDENT_JOIN_NO_ROW;
    let rightRow = RESIDENT_JOIN_NO_ROW;
    if (index < leftTotal) {
      leftRow = upperBoundInclusive(leftInclusive, index);
      if (matches[leftRow])
        rightRow =
          sortedRows[lowerBound[leftRow] + (index - (leftInclusive[leftRow] - emit[leftRow]))];
    } else {
      rightRow = upperBoundInclusive(unmatchedInclusive, index - leftTotal);
    }
    leftRows[index] = leftRow;
    rightRows[index] = rightRow;
  }
  return {leftRows, rightRows, rowCount, status};
}

/** Inclusive prefix sums as exact numbers; the GPU keeps the 64-bit total in low/high words. */
function inclusiveScan(values: ArrayLike<number>): number[] {
  const result: number[] = [];
  let total = 0;
  for (let index = 0; index < values.length; index++) {
    total += values[index];
    result.push(total);
  }
  return result;
}
/** First index whose inclusive prefix exceeds the target (the owner of an output row). */
function upperBoundInclusive(inclusive: readonly number[], target: number): number {
  let low = 0;
  let high = inclusive.length;
  while (low < high) {
    const middle = low + ((high - low) >>> 1);
    if (inclusive[middle] > target) high = middle;
    else low = middle + 1;
  }
  return low;
}
