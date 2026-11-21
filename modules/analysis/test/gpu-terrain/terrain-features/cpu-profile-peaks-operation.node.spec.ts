// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeProfilePeaksOnCPU,
  CPUProfilePeaksOperation
} from '../../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';
import {
  GPUProfilePeaksOperation,
  type ProfilePeaksOperationProps
} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks-operation';
import {getGPUProfilePeaksParameterValues} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks';
import {expectSingleProfilePeak} from './terrain-terminal-assertions';

const PARAMETERS = getGPUProfilePeaksParameterValues({minimumProminence: 1});

it('TWN-09 accepts a prominence equal to the inclusive minimum at float32 precision', () => {
  // Exact prominence is 1 - 2^-26, less than half an f32 ulp below 1, so it rounds to 1.
  const result = computeProfilePeaksOnCPU(
    Float32Array.of(2 ** -26, 1, 2 ** -26),
    Uint32Array.of(0, 3),
    PARAMETERS,
    {window: 1, minimumSide: 1, capacity: 1}
  );
  expectSingleProfilePeak({...result, mask: result.peakMask}, [0, 1, 0]);
});

it('TWN-09 keeps the earliest row when prominence ties only after float32 rounding', () => {
  // Row 1 has exact prominence 1 - 2^-26 and row 4 has 1. Both publish 1;
  // symmetric neighbors leave their refined positions at 1 and 4, exactly radius 3 apart.
  const result = computeProfilePeaksOnCPU(
    Float32Array.of(2 ** -26, 1, 2 ** -26, 0, 1, 0),
    Uint32Array.of(0, 6),
    getGPUProfilePeaksParameterValues({minimumProminence: 0}),
    {window: 1, minimumSide: 1, suppressionRadius: 3, capacity: 1}
  );
  expectSingleProfilePeak({...result, mask: result.peakMask}, [0, 1, 0, 0, 0, 0]);
});

it('N3 float64 refined positions can leave a pair outside a rounded float32 suppression radius', () => {
  // Both prominences are exactly 2. Row 1 refines to 1; at row 5, the vertex offset
  // is (0.5 * (0 - 2)) / (0 - 2*4 + 2) = 1/6. Their exact distance 4 + 1/6
  // exceeds its binary32 rounding 4.166666507720947. The CPU therefore keeps both.
  const result = computeProfilePeaksOnCPU(
    Float32Array.of(0, 2, 0, 0, 0, 4, 2, 0),
    Uint32Array.of(0, 8),
    PARAMETERS,
    {window: 1, minimumSide: 1, capacity: 2, suppressionRadius: Math.fround(4 + 1 / 6)}
  );
  expect(Array.from(result.peakMask)).toEqual([0, 1, 0, 0, 0, 1, 0, 0]);
  expect(Array.from(result.count)).toEqual([2]);
  expect(Array.from(result.ids)).toEqual([1, 5]);
  expect(Array.from(result.prominence.filter(Number.isFinite))).toEqual([2, 2]);
});

it('normalizes a circular index that rounds to the profile length', () => {
  const result = computeProfilePeaksOnCPU(
    new Float32Array([1, 0, 0, 0, 1e-7]),
    new Uint32Array([0, 5]),
    getGPUProfilePeaksParameterValues({minimumProminence: 0}),
    {window: 2, minimumSide: 1, suppressionRadius: 0, wrap: true, capacity: 1}
  );
  // The vertex offset is approximately -2.5e-8. Its wrapped index rounds to 5 in
  // float32, which represents the same circular position as 0 and must be normalized.
  expect(result.refinedIndex[0]).toBe(0);
  expect(Array.from(result.ids)).toEqual([0]);
  expect(result.count[0]).toBe(1);
  expect(result.refinedIndex[0]).toBeLessThan(5);
});

it('suppresses equal-ranked candidates to IDs 2 and 8 without dropping compact rows', () => {
  const result = computeProfilePeaksOnCPU(
    new Float32Array([0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0]),
    new Uint32Array([0, 11]),
    PARAMETERS,
    {window: 2, minimumSide: 1, suppressionRadius: 3, capacity: 2}
  );
  // Row 2 wins the tie and suppresses row 5 at the inclusive radius; row 8 survives.
  expect(Array.from(result.ids)).toEqual([2, 8]);
  expect(result.count[0]).toBe(2);
  expect(result.requiredCount[0]).toBe(2);
  expect(result.overflow[0]).toBe(0);
});

it('keeps nearby equal-height peaks belonging to different profiles', () => {
  const result = computeProfilePeaksOnCPU(
    new Float32Array([0, 0, 5, 0, 0, 5, 0, 0]),
    new Uint32Array([0, 4, 8]),
    PARAMETERS,
    {window: 2, minimumSide: 1, suppressionRadius: 3, capacity: 2}
  );
  // Global rows 2 and 5 are three samples apart, but suppression is profile-local.
  expect(Array.from(result.ids)).toEqual([2, 5]);
  expect(Array.from(result.refinedIndex.filter(Number.isFinite))).toEqual([2, 1]);
  expect(result.count[0]).toBe(2);
});

it('recovers a hand-computed parabola vertex and prominence', () => {
  // y = 4 - (x - 2.25)^2: vertex (2.25, 4), center height 3.9375.
  const values = Float32Array.from([0, 1, 2, 3, 4], index => 4 - (index - 2.25) ** 2);
  const result = computeProfilePeaksOnCPU(values, new Uint32Array([0, 5]), PARAMETERS, {
    window: 2,
    minimumSide: 2,
    capacity: 1
  });
  expect(Array.from(result.peakMask)).toEqual([0, 0, 1, 0, 0]);
  expect(result.prominence[2]).toBe(3);
  expect(result.refinedIndex[2]).toBe(2.25);
  expect(result.refinedValue[2]).toBe(4);
  expect(Array.from(result.ids)).toEqual([2]);
  expect(Array.from(result.count)).toEqual([1]);
  expect(Array.from(result.requiredCount)).toEqual([1]);
  expect(Array.from(result.overflow)).toEqual([0]);
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  expect(result.prominence.filter(Number.isNaN)).toHaveLength(4);
});

it('keeps the first equal-ranked peak, uses inclusive suppression, and clamps compact output', () => {
  const values = new Float32Array([0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0]);
  const options = {window: 2, minimumSide: 1, capacity: 1, suppressionRadius: 3};
  const result = computeProfilePeaksOnCPU(values, new Uint32Array([0, 11]), PARAMETERS, options);
  expect(Array.from(result.peakMask)).toEqual([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]);
  expect(Array.from(result.ids)).toEqual([2]);
  expect(result.requiredCount[0]).toBe(2);
  expect(result.count[0]).toBe(1);
  expect(result.overflow[0]).toBe(1);
});

it('handles circular seam peaks, gaps, empty spans and unowned samples', () => {
  const result = computeProfilePeaksOnCPU(
    new Float32Array([99, 5, 0, 0, 0, 0, 99]),
    new Uint32Array([1, 1, 6]),
    PARAMETERS,
    {window: 2, minimumSide: 1, suppressionRadius: 0, wrap: true, capacity: 2}
  );
  expect(Array.from(result.peakMask)).toEqual([0, 1, 0, 0, 0, 0, 0]);
  expect(result.refinedIndex[1]).toBe(0);
  const gaps = computeProfilePeaksOnCPU(
    new Float32Array([0, 5, 0]),
    new Uint32Array([0, 3]),
    PARAMETERS,
    {window: 1, minimumSide: 1, capacity: 1},
    new Uint32Array([1, 0, 1])
  );
  expect(gaps.count[0]).toBe(0);
  expect(gaps.refinedValue.every(Number.isNaN)).toBe(true);
});

it('rejects inconsistent arrays and validates both operation declarations at construction', () => {
  const props: ProfilePeaksOperationProps = {
    id: 'profile-validation',
    values: 'values',
    offsets: 'offsets',
    parameters: 'parameters',
    sampleCount: 3,
    window: 1,
    output: {peakMask: 'mask'}
  };
  for (const Operation of [CPUProfilePeaksOperation, GPUProfilePeaksOperation]) {
    expect(() => new Operation({...props, sampleCount: 0})).toThrow(/^profile-validation/);
    expect(() => new Operation({...props, output: {peakMask: 'values'}})).toThrow(
      /^profile-validation/
    );
  }
  expect(() => new CPUProfilePeaksOperation({...props, window: 0})).toThrow(
    /profile-validation.*window/
  );
  expect(() => new CPUProfilePeaksOperation({...props, suppressionRadius: Infinity})).toThrow(
    /profile-validation.*suppressionRadius/
  );
  expect(
    () => new CPUProfilePeaksOperation({...props, window: 5000, maximumSuppressionIterations: 2000})
  ).not.toThrow();
  expect(() =>
    computeProfilePeaksOnCPU(new Float32Array(3), new Uint32Array([0, 4]), PARAMETERS, {
      window: 1,
      capacity: 1
    })
  ).toThrow(/offsets/);
  expect(() =>
    computeProfilePeaksOnCPU(
      new Float32Array(3),
      new Uint32Array([0, 3]),
      PARAMETERS,
      {window: 1, capacity: 1},
      new Uint32Array(2)
    )
  ).toThrow(/validity/);
  expect(() =>
    computeProfilePeaksOnCPU(new Float32Array(3), new Uint32Array([0, 3]), new Float32Array(), {
      window: 1,
      capacity: 1
    })
  ).toThrow(/parameters/);
});
