// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  canonicalizeResidentJoinKeys,
  planResidentJoinPairsOnGpu,
  RESIDENT_JOIN_STATUS_FLOAT_KEY,
  sortResidentJoinRightRows
} from '../../../examples/deck/kepler-compute/src/resident-join-gpu-reference';
import {
  makeResidentJoinDictionaryKeyTables,
  makeResidentJoinNumericKey,
  planResidentJoinPairs,
  shouldDropResidentJoinHighKeys,
  type ResidentJoinType
} from '../../../examples/deck/kepler-compute/src/resident-join-plan';

const joinTypes: ResidentJoinType[] = ['LEFT', 'INNER', 'FULL'];

/** Deterministic pseudo-random generator so failures reproduce. */
function makeRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function runStringJoin(
  joinType: ResidentJoinType,
  leftCodes: Uint32Array,
  leftValidity: Uint32Array | undefined,
  leftLabels: string[],
  rightCodes: Uint32Array,
  rightValidity: Uint32Array | undefined,
  rightLabels: string[]
) {
  const tables = makeResidentJoinDictionaryKeyTables(leftLabels, rightLabels);
  const leftKey = canonicalizeResidentJoinKeys({
    words: leftCodes,
    validity: leftValidity,
    mode: 'string',
    table: tables.left
  });
  const rightKey = canonicalizeResidentJoinKeys({
    words: rightCodes,
    validity: rightValidity,
    mode: 'string',
    table: tables.right
  });
  return planResidentJoinPairsOnGpu({
    joinType,
    leftKey: leftKey.key,
    leftOk: leftKey.ok,
    rightKey: rightKey.key,
    rightOk: rightKey.ok
  });
}

test.each(
  joinTypes
)('%s GPU pipeline mirror equals the dictionary oracle on random string keys', joinType => {
  const random = makeRandom(7);
  const leftLabels = ['a', 'b', '', 'c', 'd', 'a', 'only-left'];
  const rightLabels = ['d', 'c', 'a', '', 'b', 'a', 'only-right'];
  for (let trial = 0; trial < 60; trial++) {
    const leftRows = 1 + Math.floor(random() * 40);
    const rightRows = 1 + Math.floor(random() * 40);
    const leftCodes = Uint32Array.from({length: leftRows}, () =>
      Math.floor(random() * leftLabels.length)
    );
    const rightCodes = Uint32Array.from({length: rightRows}, () =>
      Math.floor(random() * rightLabels.length)
    );
    const leftValidity = Uint32Array.from({length: leftRows}, () => (random() < 0.15 ? 0 : 1));
    const rightValidity = Uint32Array.from({length: rightRows}, () => (random() < 0.15 ? 0 : 1));
    const expected = planResidentJoinPairs({
      joinType,
      leftCodes,
      leftValidity,
      leftLabels,
      rightCodes,
      rightValidity,
      rightLabels
    });
    const actual = runStringJoin(
      joinType,
      leftCodes,
      leftValidity,
      leftLabels,
      rightCodes,
      rightValidity,
      rightLabels
    );
    expect(actual.status).toBe(0);
    expect(actual.rowCount).toBe(expected.rowCount);
    expect(Array.from(actual.leftRows)).toEqual(Array.from(expected.leftRows));
    expect(Array.from(actual.rightRows)).toEqual(Array.from(expected.rightRows));
  }
});

test.each(
  joinTypes
)('%s keeps every row of a many-to-many join beyond the old 262144 row cap', joinType => {
  // 700 x 700 duplicate keys give 490000 matches plus unmatched rows, all stable and ordered.
  const leftCodes = new Uint32Array(701).fill(0);
  leftCodes[700] = 1;
  const rightCodes = new Uint32Array(701).fill(0);
  rightCodes[0] = 2;
  const actual = runStringJoin(joinType, leftCodes, undefined, ['k', 'x'], rightCodes, undefined, [
    'k',
    'y',
    'z'
  ]);
  // Right row 0 is 'z' and unmatched; 700 right rows match each of 700 left rows.
  const expected = planResidentJoinPairs({
    joinType,
    leftCodes,
    leftLabels: ['k', 'x'],
    rightCodes,
    rightLabels: ['k', 'y', 'z']
  });
  expect(actual.rowCount).toBe(expected.rowCount);
  expect(actual.rowCount).toBeGreaterThan(262_144);
  expect(actual.leftRows.subarray(0, 800)).toEqual(expected.leftRows.subarray(0, 800));
  expect(actual.rightRows.subarray(0, 800)).toEqual(expected.rightRows.subarray(0, 800));
  expect(actual.leftRows.subarray(-800)).toEqual(expected.leftRows.subarray(-800));
  expect(actual.rightRows.subarray(-800)).toEqual(expected.rightRows.subarray(-800));
});

test('valid right rows sort first and equal keys stay in source order', () => {
  const sorted = sortResidentJoinRightRows(
    Uint32Array.from([5, 3, 5, 0, 3]),
    Uint32Array.from([1, 1, 1, 0, 1])
  );
  expect(Array.from(sorted.sortedRows)).toEqual([1, 4, 0, 2, 3]);
  expect(sorted.validCount).toBe(4);
  expect(Array.from(sorted.runIndex.subarray(0, 4))).toEqual([1, 1, 2, 2]);
});

function toWords(values: (number | null)[], format: 'sint32' | 'uint32' | 'float32') {
  const words = new Uint32Array(values.length);
  const floats = new Float32Array(words.buffer);
  const signed = new Int32Array(words.buffer);
  for (const [index, value] of values.entries()) {
    if (format === 'float32') floats[index] = value ?? 0;
    else if (format === 'sint32') signed[index] = value ?? 0;
    else words[index] = value ?? 0;
  }
  return words;
}

test.each(joinTypes)('%s numeric keys across formats equal the String(value) oracle', joinType => {
  const cases: {
    left: (number | null)[];
    leftFormat: 'sint32' | 'uint32' | 'float32';
    right: (number | null)[];
    rightFormat: 'sint32' | 'uint32' | 'float32';
  }[] = [
    {
      left: [1, 2, 3, null, 2, 10, -0, 0, -5],
      leftFormat: 'sint32',
      right: [2, 2, 3, 10, 1, null, 0, 7, -5],
      rightFormat: 'float32'
    },
    {
      // Uint32 4294967295 must not alias Int32 -1 (distinct String forms).
      left: [4294967295, 2147483648, 5, 1],
      leftFormat: 'uint32',
      right: [-1, -2147483648, 5, 1],
      rightFormat: 'sint32'
    },
    {
      left: [4294967295, 5, null],
      leftFormat: 'uint32',
      right: [4294967295, 5, 5],
      rightFormat: 'uint32'
    }
  ];
  for (const {left, leftFormat, right, rightFormat} of cases) {
    const leftWords = toWords(left, leftFormat);
    const rightWords = toWords(right, rightFormat);
    const leftValidity = Uint32Array.from(left, value => (value === null ? 0 : 1));
    const rightValidity = Uint32Array.from(right, value => (value === null ? 0 : 1));
    const leftKey = makeResidentJoinNumericKey(leftWords, leftFormat, leftValidity);
    const rightKey = makeResidentJoinNumericKey(rightWords, rightFormat, rightValidity);
    const expected = planResidentJoinPairs({
      joinType,
      leftCodes: leftKey.codes,
      leftValidity,
      leftLabels: leftKey.labels,
      rightCodes: rightKey.codes,
      rightValidity,
      rightLabels: rightKey.labels
    });
    const actualLeft = canonicalizeResidentJoinKeys({
      words: leftWords,
      validity: leftValidity,
      mode: leftFormat,
      dropHighKeys: shouldDropResidentJoinHighKeys(leftFormat, rightFormat)
    });
    const actualRight = canonicalizeResidentJoinKeys({
      words: rightWords,
      validity: rightValidity,
      mode: rightFormat,
      dropHighKeys: shouldDropResidentJoinHighKeys(rightFormat, leftFormat)
    });
    const actual = planResidentJoinPairsOnGpu({
      joinType,
      leftKey: actualLeft.key,
      leftOk: actualLeft.ok,
      rightKey: actualRight.key,
      rightOk: actualRight.ok
    });
    expect(Array.from(actual.leftRows)).toEqual(Array.from(expected.leftRows));
    expect(Array.from(actual.rightRows)).toEqual(Array.from(expected.rightRows));
  }
});

test('non-integral and out-of-range Float32 keys report status, null rows are skipped', () => {
  const words = toWords([0.5, 3, 33_554_432, 1.5], 'float32');
  const result = canonicalizeResidentJoinKeys({
    words,
    validity: Uint32Array.from([1, 1, 1, 0]),
    mode: 'float32'
  });
  expect(result.status & RESIDENT_JOIN_STATUS_FLOAT_KEY).toBe(RESIDENT_JOIN_STATUS_FLOAT_KEY);
  expect(Array.from(result.ok)).toEqual([0, 1, 0, 0]);
  expect(
    canonicalizeResidentJoinKeys({words, validity: Uint32Array.from([0, 1, 0, 0]), mode: 'float32'})
      .status
  ).toBe(0);
});

test('dictionary key tables share the first right code per label and drop empties', () => {
  const tables = makeResidentJoinDictionaryKeyTables(['a', '', 'z', 'b'], ['b', 'a', 'a', '']);
  expect(Array.from(tables.left)).toEqual([1, 0xffffffff, 0xffffffff, 0]);
  expect(Array.from(tables.right)).toEqual([0, 1, 1, 0xffffffff]);
});
