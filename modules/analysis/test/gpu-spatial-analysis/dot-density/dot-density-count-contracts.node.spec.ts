// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeDotPhiloxReference, generateDotCountsOnCPU} from './dot-density-cpu';

it('the independent BigInt Philox reference matches the Random123 known-answer vector', () => {
  // Random123 tests/kat_vectors: Philox4x32, ten rounds, zero counter and zero key.
  expect(computeDotPhiloxReference([0, 0, 0, 0], [0, 0])).toEqual([
    0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8
  ]);
});

it('integer requests remain exact across counts and values routes at a float32 subtraction boundary (B2)', () => {
  // Independent Random123 draw convention: slot 1, seed 9, purpose 6, high 24 bits.
  const remainder = (computeDotPhiloxReference([1, 0, 0, 0], [9, 6])[0] >>> 8) / 2 ** 24;
  expect(remainder).toBe(0.9994140863418579);
  const parameters = new Uint32Array(8);
  parameters[0] = 9;
  new Float32Array(parameters.buffer)[1] = 1;
  const polygons = {
    positions: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 5, 0, 3, 2),
    featureOffsets: Uint32Array.of(0, 1, 2),
    polygonOffsets: Uint32Array.of(0, 1, 2),
    ringOffsets: Uint32Array.of(0, 3, 6),
    parameters,
    capacity: 16385
  };
  // Integer requests have no Bernoulli remainder. One zero-area request fails;
  // feature 1 must emit exactly 16385, even with its pinned uniform 0.9994140863418579.
  for (const source of [{counts: Uint32Array.of(1, 16385)}, {values: Float32Array.of(1, 16385)}]) {
    const result = generateDotCountsOnCPU({...polygons, ...source});
    expect(Array.from(result.slotCounts)).toEqual([0, 16385]);
    expect(Array.from(result.slotStarts)).toEqual([0, 0]);
    expect(result.count).toBe(16385);
    expect(result.requiredCount).toBe(16385);
    expect(result.failedCount).toBe(1);
    expect(result.featureIds).toHaveLength(16385);
    expect(Array.from(result.featureIds).every(identifier => identifier === 1)).toBe(true);
  }
});

it('zero-area integer requests contribute failures but no slots beside a positive-area feature (GEO-130)', () => {
  const parameters = new Uint32Array(8);
  new Float32Array(parameters.buffer)[1] = 1;
  const input = {
    // Feature 0 is collinear (area 0); feature 1 is a right triangle (area 2).
    positions: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 5, 0, 3, 2),
    featureOffsets: Uint32Array.of(0, 1, 2),
    polygonOffsets: Uint32Array.of(0, 1, 2),
    ringOffsets: Uint32Array.of(0, 3, 6),
    counts: Uint32Array.of(3, 2),
    parameters,
    capacity: 8
  };
  const result = generateDotCountsOnCPU(input);
  // By hand: three requests fail before the prefix; only feature 1's two slots remain.
  expect(Array.from(result.slotCounts)).toEqual([0, 2]);
  expect(Array.from(result.slotStarts)).toEqual([0, 0]);
  expect(Array.from(result.featureIds)).toEqual([1, 1]);
  expect(Array.from(result.categories)).toEqual([0, 0]);
  expect([result.requiredCount, result.count, result.overflow, result.candidateOverflow]).toEqual([
    2, 2, 0, 0
  ]);
  expect(result).toHaveProperty('failedCount', 3);
  // With capacity 2, the per-slot limit is 3. Nine requests clamp to 3 before area rejection.
  const clamped = generateDotCountsOnCPU({...input, counts: Uint32Array.of(9, 2), capacity: 2});
  expect(Array.from(clamped.slotCounts)).toEqual([0, 2]);
  expect(Array.from(clamped.slotStarts)).toEqual([0, 0]);
  expect(Array.from(clamped.featureIds)).toEqual([1, 1]);
  expect([
    clamped.requiredCount,
    clamped.count,
    clamped.overflow,
    clamped.candidateOverflow
  ]).toEqual([2, 2, 0, 1]);
  expect(clamped).toHaveProperty('failedCount', 3);
});

it('S1 nonunit scale has no fractional request after binary32 product rounding', () => {
  const value = Math.fround(819.15);
  expect(Math.fround(value * 20)).toBe(16383);
  expect(value * 20 - 16383).toBe(0.00048828125);
  expect((computeDotPhiloxReference([0, 0, 0, 0], [411, 6])[0] >>> 8) / 2 ** 24).toBe(
    0.0003591179847717285
  );
  const parameters = new Uint32Array(8);
  parameters[0] = 411;
  new Float32Array(parameters.buffer)[1] = 20;
  const result = generateDotCountsOnCPU({
    positions: Float32Array.of(0, 0, 2, 0, 0, 2),
    featureOffsets: Uint32Array.of(0, 1),
    polygonOffsets: Uint32Array.of(0, 1),
    ringOffsets: Uint32Array.of(0, 3),
    values: Float32Array.of(value),
    parameters,
    capacity: 16383
  });
  expect([result.count, result.requiredCount, result.candidateOverflow]).toEqual([16383, 16383, 0]);
});

it('S3 positive infinite scaled requests use the preserved per-slot clamp path', () => {
  const parameters = new Uint32Array(8);
  new Float32Array(parameters.buffer)[1] = Infinity;
  const result = generateDotCountsOnCPU({
    positions: Float32Array.of(0, 0, 2, 0, 0, 2),
    featureOffsets: Uint32Array.of(0, 1),
    polygonOffsets: Uint32Array.of(0, 1),
    ringOffsets: Uint32Array.of(0, 3),
    values: Float32Array.of(2),
    parameters,
    capacity: 3
  });
  // One slot reserves capacity+1 = 4 as a loss witness, then final capacity retains 3 dots.
  expect([result.count, result.requiredCount, result.candidateOverflow]).toEqual([3, 4, 1]);
});
