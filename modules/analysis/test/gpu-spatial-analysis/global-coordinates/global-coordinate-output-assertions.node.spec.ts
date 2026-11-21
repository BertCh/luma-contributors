// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeSphericalArcOverlayOnCPU} from '../../../src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation';
import {ARC_SCALING_CASES, getArcScalingOperand} from './global-coordinate-scaling-fixtures';
import {
  expectArcScalingOutput,
  expectDatumSelectionPositions,
  expectDatumGridCountPositions,
  expectConvexReadbackLengths,
  getDatumSourceObservation
} from './global-coordinate-output-assertions';

it('rejects stale unrotated arc geometry after the rotated upload', () => {
  const reference = ARC_SCALING_CASES[0];
  const operand = (right: boolean) => ({
    positions: Float32Array.from(getArcScalingOperand(8, right).flat()),
    ringOffsets: Uint32Array.of(0, 4),
    referenceInside: true
  });
  const result = computeSphericalArcOverlayOnCPU(
    {left: operand(false), right: operand(true)},
    {
      referencePoint: [0, 0, 1],
      operation: 'intersection',
      capacity: 256,
      ringCapacity: 64
    }
  );
  const output = {
    positions: Array.from(result.positions),
    rings: Array.from(result.ringOffsets),
    counts: Array.from(result.counts),
    status: Array.from(result.status)
  };
  expect(() => expectArcScalingOutput(output, reference, false)).not.toThrow();
  expect(() => expectArcScalingOutput(output, reference, true)).toThrow();
  const rotated = {
    ...output,
    positions: output.positions.map((value, index, values) =>
      index % 3 === 0
        ? value * Math.cos(0.1) - values[index + 1] * Math.sin(0.1)
        : index % 3 === 1
          ? values[index - 1] * Math.sin(0.1) + value * Math.cos(0.1)
          : value
    )
  };
  expect(() => expectArcScalingOutput(rotated, reference, true)).not.toThrow();
});
it('rejects empty and truncated datum selection readbacks', () => {
  expect(() => expectDatumSelectionPositions([], false, true)).toThrow();
  expect(() => expectDatumSelectionPositions([NaN, NaN, NaN], false, true)).toThrow();
  expect(() =>
    expectDatumSelectionPositions([NaN, NaN, NaN, NaN, NaN, NaN], false, true)
  ).not.toThrow();
});
it('rejects empty and truncated datum benchmark readbacks', () => {
  expect(() => expectDatumGridCountPositions([], 8, true)).toThrow();
  expect(() => expectDatumGridCountPositions(new Array(11).fill(NaN), 8, true)).toThrow();
  expect(() => expectDatumGridCountPositions(new Array(12).fill(NaN), 8, true)).not.toThrow();
});
it('requires full convex position and measure readbacks', () => {
  expect(() => expectConvexReadbackLengths([], [])).toThrow();
  expect(() => expectConvexReadbackLengths(new Array(47).fill(0), new Array(8).fill(0))).toThrow();
  expect(() => expectConvexReadbackLengths(new Array(48).fill(0), new Array(7).fill(0))).toThrow();
});
it('records bounded table-driven datum shader sizes without requiring source growth', () => {
  expect(getDatumSourceObservation('fn main() {}', 512)).toEqual({
    gridCount: 512,
    generatedBytes: 12
  });
});
