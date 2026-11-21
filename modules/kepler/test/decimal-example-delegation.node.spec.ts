// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as library from '@luma-contributors/analysis/dataframe-analysis';
import * as exampleAggregate from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-aggregate';
import * as exampleDecimalReference from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-decimal-reference';
import * as exampleInt64Filter from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-int64-filter';
import * as exampleNarrowing from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-narrowing';
import * as exampleReference from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-reference';
import * as exampleRunner from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-runner';
import * as exampleWgsl from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-wgsl';

import * as arrow from 'apache-arrow';
import * as exampleDecimalAggregate from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-decimal-aggregate';
import {makeDecimal128WordVectors} from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-decimal-aggregate';
import {createWideNumericUploadDevice} from '../../analysis/test/utils/wide-numeric-upload-test-utils';

test('SEAM-114: example uploads the two sliced Decimal128 rows exactly once and preserves caller ownership', () => {
  // Little-endian Decimal128 of positive integers has the integer in word 0 and three zero words.
  const column = new arrow.Vector([
    arrow.makeData({
      type: new arrow.Decimal(0, 38, 128),
      length: 3,
      data: new Uint32Array([11, 0, 0, 0, 22, 0, 0, 0, 33, 0, 0, 0]),
      nullCount: 0
    })
  ]).slice(1, 3);
  const {device, buffers} = createWideNumericUploadDevice();
  const uploaded = makeDecimal128WordVectors(device, column);
  try {
    expect(uploaded.words.length).toBe(2);
    expect(buffers).toHaveLength(1);
    expect(uploaded.validity).toBeUndefined();
    expect(uploaded.unusedValidity).toBeUndefined();
    expect(Array.from(buffers[0].data)).toEqual([22, 0, 0, 0, 33, 0, 0, 0]);
    expect(buffers.every(buffer => buffer.destroyCount === 0)).toBe(true);
  } finally {
    // Existing example callers own these vectors and can still release them separately.
    uploaded.words.destroy();
    uploaded.validity?.destroy();
    uploaded.unusedValidity?.destroy();
  }
  expect(buffers.map(buffer => buffer.destroyCount)).toEqual(buffers.map(() => 1));
});

test('DPC-126: example sliced Decimal128 validity uses the logical bitmap offset and exact little-endian words', () => {
  // Slice rows 1..2: bit 1 is absent, bit 2 is present. Word patterns are literal
  // little-endian signed 128-bit values; -1 has all four words set.
  const column = new arrow.Vector([
    arrow.makeData({
      type: new arrow.Decimal(0, 38, 128),
      length: 3,
      data: Uint32Array.of(
        11,
        0,
        0,
        0,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0x89abcdef,
        0x01234567,
        0,
        0
      ),
      nullBitmap: Uint8Array.of(0b101),
      nullCount: 1
    })
  ]).slice(1, 3);
  const {device, buffers} = createWideNumericUploadDevice();
  const uploaded = makeDecimal128WordVectors(device, column);
  expect(buffers).toHaveLength(2);
  expect(Array.from(buffers[0].data)).toEqual([
    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x89abcdef, 0x01234567, 0, 0
  ]);
  expect(Array.from(buffers[1].data)).toEqual([0, 1]);
  expect(buffers.every(buffer => buffer.destroyCount === 0)).toBe(true);
  uploaded.destroy();
  uploaded.destroy();
  expect(buffers.map(buffer => buffer.destroyCount)).toEqual([1, 1]);
});

test('DPC-126: example failed validity allocation cleans up the words exactly once', () => {
  const column = new arrow.Vector([
    arrow.makeData({
      type: new arrow.Decimal(0, 38, 128),
      length: 2,
      data: Uint32Array.of(11, 0, 0, 0, 22, 0, 0, 0),
      nullBitmap: Uint8Array.of(1),
      nullCount: 1
    })
  ]);
  for (const allocation of [1, 2]) {
    const {device, buffers, failure} = createWideNumericUploadDevice(allocation);
    expect(() => makeDecimal128WordVectors(device, column)).toThrow(failure);
    expect(buffers).toHaveLength(allocation - 1);
    expect(buffers.map(buffer => buffer.destroyCount)).toEqual(allocation === 1 ? [] : [1]);
  }
});

test('DPC-126 S3: compatibility entries retain their exact runtime export names', () => {
  const entries: [Record<string, unknown>, string[]][] = [
    [exampleAggregate, ['aggregateInteger64OnGpu']],
    [
      exampleDecimalAggregate,
      [
        'aggregateDecimal128Column',
        'aggregateDecimal128OnGpu',
        'makeDecimal128SelectionExpression',
        'makeDecimal128WordVectors'
      ]
    ],
    [
      exampleDecimalReference,
      [
        'DECIMAL128_MAXIMUM',
        'DECIMAL128_MINIMUM',
        'splitDecimal128ToWords',
        'joinWordsToDecimal128',
        'combineDecimal128SumAccumulators',
        'summarizeDecimal128Reference',
        'formatDecimal128',
        'normalizeDecimal128Comparison',
        'compareDecimal128Reference'
      ]
    ],
    [
      exampleInt64Filter,
      [
        'resolveInteger64Range',
        'makeInteger64RangeSelectionExpression',
        'aggregateInteger64Range',
        'isInteger64InRange'
      ]
    ],
    [exampleNarrowing, ['scanWideNumericColumnOnGpu']],
    [
      exampleReference,
      [
        'splitBigIntToWords',
        'joinWordsToBigInt',
        'splitDoubleToWords',
        'addWords',
        'subtractWords',
        'negateWords',
        'isUnsignedLess',
        'isSignedLess',
        'multiplyWords32',
        'multiplyWordsLow',
        'multiplyWords128',
        'shiftWordsRight',
        'shiftWordsLeft',
        'countLeadingZeroWords',
        'divideWordsBySmall',
        'divideSignedWordsBySmall',
        'convertWordsToFloat32',
        'sumWordsWithCarry',
        'combineWordSumAccumulators',
        'classifyWordsNarrowing',
        'decodeNarrowingError',
        'roundBigIntToFloat32',
        'summarizeInteger64Reference',
        'summarizeNarrowingReference',
        'WORD_RADIX'
      ]
    ],
    [exampleRunner, ['runWideNumericKernel']],
    [exampleWgsl, ['WIDE_NUMERIC_WGSL_HELPERS']]
  ];
  for (const [adapter, names] of entries) {
    expect(Object.keys(adapter).sort()).toEqual([...names].sort());
    for (const name of names) {
      if (
        [
          'makeDecimal128WordVectors',
          'summarizeNarrowingReference',
          'scanWideNumericColumnOnGpu'
        ].includes(name)
      )
        continue;
      expect(adapter[name], name).toBe((library as unknown as Record<string, unknown>)[name]);
    }
  }
});

test('DPC-126 B5: compatibility narrowing reports retain number errors and JSON serialization', async () => {
  // Nearest-even f32 rounds 2^24+1 to 2^24: the legacy report error is number 1.
  const reference = exampleReference.summarizeNarrowingReference('uint64', [16777217n]);
  expect(reference.maximumAbsoluteError).toBe(1);
  expect(() => JSON.stringify(reference)).not.toThrow();
  const spy = vi.spyOn(library, 'scanWideNumericColumnOnGpu').mockResolvedValue({
    ...reference,
    maximumAbsoluteError: 1n
  });
  try {
    const scanned = await exampleNarrowing.scanWideNumericColumnOnGpu(
      {} as Parameters<typeof exampleNarrowing.scanWideNumericColumnOnGpu>[0]
    );
    expect(scanned.maximumAbsoluteError).toBe(1);
    expect(() => JSON.stringify(scanned)).not.toThrow();
  } finally {
    spy.mockRestore();
  }
});

test('DPC-126 S4: example runner inherits positive resultWordCount validation before allocation', async () => {
  const {device, buffers} = createWideNumericUploadDevice();
  await expect(
    exampleRunner.runWideNumericKernel({
      device,
      id: 'legacy-runner-validation',
      resultWordCount: 0,
      phaseBodies: [],
      words: undefined as unknown as Parameters<
        typeof exampleRunner.runWideNumericKernel
      >[0]['words']
    })
  ).rejects.toThrow(/resultWordCount/);
  expect(buffers).toEqual([]);
});
