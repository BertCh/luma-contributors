// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUPermutationParameterValues,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  getPhilox4x32,
  PhiloxStream
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-random';

const hex = (words: number[]) => words.map(word => word.toString(16).padStart(8, '0')).join(' ');

it('Philox 4x32-10 matches the Random123 known-answer vectors', () => {
  expect(hex(getPhilox4x32([0, 0, 0, 0], [0, 0]))).toBe('6627e8d5 e169c58d bc57ac4c 9b00dbd8');
  const ones = 0xffffffff;
  expect(hex(getPhilox4x32([ones, ones, ones, ones], [ones, ones]))).toBe(
    '408f276d 41c83b0e a20bc7c6 6d5451fd'
  );
  expect(
    hex(getPhilox4x32([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], [0xa4093822, 0x299f31d0]))
  ).toBe('d16cfe09 94fdcceb 5001e420 24126ea1');
});

it('PhiloxStream walks blocks in order and nextBelow is unbiased and in range', () => {
  const stream = new PhiloxStream([5, 6], 7, 8, 9);
  const first = [
    stream.nextUint32(),
    stream.nextUint32(),
    stream.nextUint32(),
    stream.nextUint32()
  ];
  expect(first).toEqual(getPhilox4x32([0, 7, 8, 9], [5, 6]));
  expect(stream.nextUint32()).toBe(getPhilox4x32([1, 7, 8, 9], [5, 6])[0]);
  for (const bound of [1, 2, 3, 7, 1000, 0x80000001, 0xffffffff]) {
    const draws = new PhiloxStream([1, 2], bound, 0, 0);
    for (let index = 0; index < 2000; index++) {
      const value = draws.nextBelow(bound);
      expect(Number.isInteger(value) && value >= 0 && value < bound).toBe(true);
    }
  }
  const counts = new Array(6).fill(0);
  const dice = new PhiloxStream([3, 4], 0, 0, 0);
  for (let index = 0; index < 60000; index++) {
    counts[dice.nextBelow(6)]++;
  }
  for (const count of counts) {
    // Six standard deviations of a binomial(60000, 1/6).
    expect(Math.abs(count - 10000)).toBeLessThan(6 * Math.sqrt(60000 * (1 / 6) * (5 / 6)));
  }
});

it('getGPUPermutationParameterValues packs the seed words, count and level bits', () => {
  const values = getGPUPermutationParameterValues({
    seed: 2 ** 40 + 5,
    permutationCount: 999,
    significanceLevel: 0.01
  });
  expect(values.length).toBe(GPU_PERMUTATION_PARAMETER_LENGTH);
  expect(values[0]).toBe(5);
  expect(values[1]).toBe(256);
  expect(values[2]).toBe(999);
  expect(new Float32Array(values.buffer)[3]).toBe(Math.fround(0.01));
  expect(() => getGPUPermutationParameterValues({seed: -1, permutationCount: 9})).toThrow(/seed/);
  expect(() => getGPUPermutationParameterValues({seed: 1, permutationCount: 0})).toThrow(
    /permutationCount/
  );
});
