// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getBinarySearchWGSL} from '../../src/analysis-core/kernels/binary-search-wgsl';
import {UINT64_WGSL} from '../../src/analysis-core/kernels/uint64-wgsl';
import {
  DECODE_ORDERED_KEY_WGSL,
  GET_ORDERED_KEY_WGSL,
  IS_FINITE_VALUE_WGSL,
  IS_NAN_VALUE_WGSL
} from '../../src/analysis-core/kernels/wgsl-common';
import {requireTestDevice} from './require-test-device';
import {createSeededRandom, getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// The float tests, ordered keys, binary search and 64-bit word modules of analysis-core/kernels
// against CPU references (bit patterns, numeric order, BigInt).

const SPECIAL_BITS = [
  0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x00000001, 0x807fffff, 0x7f7fffff, 0xff7fffff,
  0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7f800001, 0xff800001, 0x7fffffff, 0x7fa00000
];

function isNaNBits(bits: number): boolean {
  return (bits & 0x7fffffff) >>> 0 > 0x7f800000;
}

function getOrderedKey(bits: number): number {
  return (bits & 0x80000000 ? ~bits : bits ^ 0x80000000) >>> 0;
}

function toBits(value: number): number {
  return new Uint32Array(Float32Array.of(value).buffer)[0];
}

it('isFiniteValue and isNanValue classify every special bit pattern', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const bits = Uint32Array.from(SPECIAL_BITS);
  // The same patterns through an f32 binding (as a contributor reads them) and through bitcast.
  const words = await runWGSLModuleKernel(device, {
    id: 'float-tests',
    inputs: {bits, floats: new Float32Array(bits.buffer.slice(0))},
    outputLength: bits.length,
    invocationCount: bits.length,
    declarations: `${IS_FINITE_VALUE_WGSL}${IS_NAN_VALUE_WGSL}`,
    body: `let fromBits = bitcast<f32>(bits[bitsOffset + index]);
  let fromStorage = floats[floatsOffset + index];
  output[outputOffset + index] = select(0u, 1u, isFiniteValue(fromBits)) | select(0u, 2u, isNanValue(fromBits)) |
    select(0u, 4u, isFiniteValue(fromStorage)) | select(0u, 8u, isNanValue(fromStorage));`
  });
  const expected = SPECIAL_BITS.map(value => {
    const isNan = isNaNBits(value);
    const isFiniteBits = (value & 0x7f800000) >>> 0 !== 0x7f800000;
    return (isFiniteBits ? 5 : 0) | (isNan ? 10 : 0);
  });
  expect(Array.from(words)).toEqual(expected);
});

it('ordered keys sort floats numerically and decode to the same bits', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(41);
  const values = [
    ...SPECIAL_BITS.filter(value => !isNaNBits(value)),
    ...Array.from({length: 200}, () => toBits((random() - 0.5) * 10 ** (20 * random() - 10)))
  ];
  const bits = Uint32Array.from(values);
  const words = await runWGSLModuleKernel(device, {
    id: 'ordered-keys',
    inputs: {bits},
    outputLength: bits.length * 2,
    invocationCount: bits.length,
    declarations: `${GET_ORDERED_KEY_WGSL}${DECODE_ORDERED_KEY_WGSL}`,
    body: `let key = getOrderedKey(bitcast<f32>(bits[bitsOffset + index]));
  output[outputOffset + 2u * index] = key;
  output[outputOffset + 2u * index + 1u] = bitcast<u32>(decodeOrderedKey(key));`
  });
  const keys = values.map((_, index) => words[2 * index]);
  expect(keys).toEqual(values.map(getOrderedKey));
  expect(values.map((_, index) => words[2 * index + 1])).toEqual(values);
  const floats = new Float32Array(bits.buffer);
  const byKey = values.map((_, index) => index).sort((left, right) => keys[left] - keys[right]);
  for (let rank = 1; rank < byKey.length; rank++) {
    const previous = floats[byKey[rank - 1]];
    const current = floats[byKey[rank]];
    expect(previous <= current, `${previous} before ${current}`).toBe(true);
  }
});

function getCPUBound<T>(
  sorted: readonly T[],
  first: number,
  last: number,
  isLeft: (value: T) => boolean
): number {
  let index = first;
  while (index < last && isLeft(sorted[index])) {
    index++;
  }
  return Math.max(index, first);
}

it('lowerBound and upperBound match a linear scan for u32, f32 and u64 keys', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(42);
  const unsigned = Array.from({length: 97}, () => Math.floor(random() * 40) * 0x01000000).sort(
    (left, right) => left - right
  );
  unsigned.push(0xffffffff);
  const floatValues = [-Infinity, -2.5, -0, 0, 0, 1e-30, 1, 1, 1, 3.5, Infinity];
  const wide = Array.from(
    {length: 64},
    () => (BigInt(Math.floor(random() * 6)) << 32n) | BigInt(Math.floor(random() * 4) * 0x40000000)
  ).sort((left, right) => (left < right ? -1 : left > right ? 1 : 0));
  const wideWords = new Uint32Array(wide.length * 2);
  wide.forEach((value, index) => {
    wideWords[2 * index] = Number(value & 0xffffffffn);
    wideWords[2 * index + 1] = Number(value >> 32n);
  });
  // Query: [kind, first, last, key low, key high]. Kind 0 u32, 1 f32 (key as bits), 2 u64.
  const queries: number[][] = [];
  for (let repeat = 0; repeat < 200; repeat++) {
    const first = Math.floor(random() * 10);
    const last = first + Math.floor(random() * 90);
    queries.push([0, first, last, (Math.floor(random() * 42) * 0x01000000) >>> 0, 0]);
  }
  queries.push([0, 0, unsigned.length, 0xffffffff, 0], [0, 5, 3, 7, 0], [0, 4, 4, 0, 0]);
  // Reversed ranges (`first > last`) of every key type: documented to return `first`.
  const reversedQueries = [
    [0, 5, 3, 7, 0],
    [1, 6, 2, toBits(1), 0],
    [2, 9, 4, 0, 1]
  ];
  queries.push(...reversedQueries.slice(1));
  for (const key of [-Infinity, -3, -0, 0, 1e-31, 1, 2, 3.5, Infinity]) {
    queries.push([1, 0, floatValues.length, toBits(key), 0], [1, 2, 7, toBits(key), 0]);
  }
  for (let repeat = 0; repeat < 100; repeat++) {
    const first = Math.floor(random() * 8);
    queries.push([
      2,
      first,
      first + Math.floor(random() * 56),
      Math.floor(random() * 4) * 0x40000000,
      Math.floor(random() * 7)
    ]);
  }
  const words = await runWGSLModuleKernel(device, {
    id: 'binary-search',
    inputs: {
      unsignedKeys: Uint32Array.from(unsigned),
      floatKeys: Float32Array.from(floatValues),
      wideKeys: wideWords,
      queries: Uint32Array.from(queries.flat())
    },
    outputLength: queries.length * 2,
    invocationCount: queries.length,
    declarations: `${getBinarySearchWGSL('unsignedKeys')}
${getBinarySearchWGSL('floatKeys', 'f32')}
${getBinarySearchWGSL('wideKeys', 'u64')}`,
    body: `let base = queriesOffset + index * 5u;
  let kind = queries[base];
  let first = queries[base + 1u];
  let last = queries[base + 2u];
  var bounds = vec2<u32>(0u, 0u);
  if (kind == 0u) {
    bounds = vec2<u32>(lowerBoundUnsignedKeys(first, last, queries[base + 3u]), upperBoundUnsignedKeys(first, last, queries[base + 3u]));
  } else if (kind == 1u) {
    let key = bitcast<f32>(queries[base + 3u]);
    bounds = vec2<u32>(lowerBoundFloatKeys(first, last, key), upperBoundFloatKeys(first, last, key));
  } else {
    // In the shader a 64-bit key is (high, low); the query row stores low then high.
    let key = vec2<u32>(queries[base + 4u], queries[base + 3u]);
    bounds = vec2<u32>(lowerBoundWideKeys(first, last, key), upperBoundWideKeys(first, last, key));
  }
  output[outputOffset + 2u * index] = bounds.x;
  output[outputOffset + 2u * index + 1u] = bounds.y;`
  });
  const floatOrder = (value: number): number => getOrderedKey(toBits(value));
  const mismatches: string[] = [];
  queries.forEach(([kind, first, last, low, high], index) => {
    let expected: [number, number];
    if (kind === 0) {
      expected = [
        getCPUBound(unsigned, first, last, value => value < low),
        getCPUBound(unsigned, first, last, value => value <= low)
      ];
    } else if (kind === 1) {
      const key = getOrderedKey(low);
      expected = [
        getCPUBound(floatValues, first, last, value => floatOrder(value) < key),
        getCPUBound(floatValues, first, last, value => floatOrder(value) <= key)
      ];
    } else {
      const key = (BigInt(high) << 32n) | BigInt(low);
      expected = [
        getCPUBound(wide, first, last, value => value < key),
        getCPUBound(wide, first, last, value => value <= key)
      ];
    }
    const actual = [words[2 * index], words[2 * index + 1]];
    if (actual[0] !== expected[0] || actual[1] !== expected[1]) {
      mismatches.push(`query ${queries[index].join(',')}: ${actual} expected ${expected}`);
    }
  });
  expect(mismatches.slice(0, 10)).toEqual([]);
  // Pin the reversed-range result explicitly (the linear-scan reference clamps to `first` too).
  for (const reversed of reversedQueries) {
    const index = queries.findIndex(query => query.join(',') === reversed.join(','));
    const first = reversed[1];
    expect([words[2 * index], words[2 * index + 1]], `query ${reversed.join(',')}`).toEqual([
      first,
      first
    ]);
  }
});

const MASK_64 = (1n << 64n) - 1n;

function toSigned(value: bigint): bigint {
  return value >= 1n << 63n ? value - (1n << 64n) : value;
}

it('64-bit word arithmetic matches BigInt', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(43);
  const randomWord = (): number => {
    const choice = random();
    // Mix edge words with uniform ones so carries and borrows are exercised.
    if (choice < 0.15) return 0;
    if (choice < 0.3) return 0xffffffff;
    if (choice < 0.4) return 0x80000000;
    return Math.floor(random() * 4294967296);
  };
  const caseCount = 512;
  // Per case: left low, left high, right low, right high, shift, divisor.
  const inputs = new Uint32Array(caseCount * 6);
  for (let index = 0; index < caseCount; index++) {
    inputs.set(
      [
        randomWord(),
        randomWord(),
        randomWord(),
        randomWord(),
        Math.floor(random() * 70),
        index % 7 === 0 ? 1 : randomWord() || 3
      ],
      index * 6
    );
  }
  const OUTPUTS = 22;
  const words = await runWGSLModuleKernel(device, {
    id: 'uint64-words',
    inputs: {inputs},
    outputLength: caseCount * OUTPUTS,
    invocationCount: caseCount,
    declarations: UINT64_WGSL,
    body: `let base = inputsOffset + index * 6u;
  // UINT64_WGSL values are (high, low) in the shader; the case row stores low then high.
  let left = vec2<u32>(inputs[base + 1u], inputs[base]);
  let right = vec2<u32>(inputs[base + 3u], inputs[base + 2u]);
  let bits = inputs[base + 4u];
  let divisor = inputs[base + 5u];
  let outputBase = outputOffset + index * ${OUTPUTS}u;
  let sum = addUint64(left, right);
  let difference = subtractUint64(left, right);
  let negated = negateInt64(left);
  let shiftedLeft = shiftLeftUint64(left, bits);
  let shiftedRight = shiftRightUint64(left, bits);
  let wide = multiplyUint32Wide(left.y, right.y);
  let byWord = multiplyUint64ByUint32(left, right.y);
  let product = multiplyUint64(left, right);
  let division = divideUint64ByUint32(left, divisor);
  output[outputBase] = sum.x; output[outputBase + 1u] = sum.y;
  output[outputBase + 2u] = difference.x; output[outputBase + 3u] = difference.y;
  output[outputBase + 4u] = negated.x; output[outputBase + 5u] = negated.y;
  output[outputBase + 6u] = shiftedLeft.x; output[outputBase + 7u] = shiftedLeft.y;
  output[outputBase + 8u] = shiftedRight.x; output[outputBase + 9u] = shiftedRight.y;
  output[outputBase + 10u] = wide.x; output[outputBase + 11u] = wide.y;
  output[outputBase + 12u] = byWord.x; output[outputBase + 13u] = byWord.y;
  output[outputBase + 14u] = product.x; output[outputBase + 15u] = product.y;
  output[outputBase + 16u] = division.quotient.x; output[outputBase + 17u] = division.quotient.y;
  output[outputBase + 18u] = division.remainder;
  output[outputBase + 19u] = bitcast<u32>(compareUint64(left, right) * 4i + compareInt64(left, right)) ^ select(0u, 0x100u, isLessUint64(left, right)) ^ select(0u, 0x200u, isLessOrEqualUint64(left, right));
  output[outputBase + 20u] = bitcast<u32>(uint64ToFloat(left));
  output[outputBase + 21u] = bitcast<u32>(int64ToFloat(left));`
  });
  const floats = getFloatWords(words);
  const mismatches: string[] = [];
  const pair = (low: number, high: number): bigint => (BigInt(high) << 32n) | BigInt(low);
  for (let index = 0; index < caseCount; index++) {
    const [leftLow, leftHigh, rightLow, rightHigh, bits, divisor] = inputs.slice(
      index * 6,
      index * 6 + 6
    );
    const left = pair(leftLow, leftHigh);
    const right = pair(rightLow, rightHigh);
    const base = index * OUTPUTS;
    // Results are written `.x` then `.y`, which is high then low.
    const read = (slot: number): bigint => pair(words[base + slot + 1], words[base + slot]);
    const compare = (first: bigint, second: bigint): number =>
      first < second ? -1 : first > second ? 1 : 0;
    const expected: [string, bigint, bigint][] = [
      ['add', read(0), (left + right) & MASK_64],
      ['subtract', read(2), (left - right) & MASK_64],
      ['negate', read(4), -left & MASK_64],
      ['shift left', read(6), bits >= 64 ? 0n : (left << BigInt(bits)) & MASK_64],
      ['shift right', read(8), bits >= 64 ? 0n : left >> BigInt(bits)],
      ['wide', read(10), BigInt(leftLow) * BigInt(rightLow)],
      ['by word', read(12), (left * BigInt(rightLow)) & MASK_64],
      ['product', read(14), (left * right) & MASK_64],
      ['quotient', read(16), left / BigInt(divisor)],
      ['remainder', BigInt(words[base + 18]), left % BigInt(divisor)]
    ];
    for (const [name, actual, wanted] of expected) {
      if (actual !== wanted) {
        mismatches.push(`case ${index} ${name}: ${actual} expected ${wanted}`);
      }
    }
    const comparison =
      ((compare(left, right) * 4 + compare(toSigned(left), toSigned(right))) >>> 0) ^
      (left < right ? 0x100 : 0) ^
      (left <= right ? 0x200 : 0);
    if (words[base + 19] !== comparison >>> 0) {
      mismatches.push(`case ${index} compare: ${words[base + 19]} expected ${comparison >>> 0}`);
    }
    for (const [slot, value] of [
      [20, Number(left)],
      [21, Number(toSigned(left))]
    ] as const) {
      if (Math.abs(floats[base + slot] - value) > 2.5e-7 * Math.abs(value)) {
        mismatches.push(`case ${index} float ${slot}: ${floats[base + slot]} expected ${value}`);
      }
    }
  }
  expect(mismatches.slice(0, 10)).toEqual([]);
});
