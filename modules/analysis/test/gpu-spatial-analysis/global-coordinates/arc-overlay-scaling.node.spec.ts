// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeSphericalArcOverlayOnCPU} from '../../../src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation';
import {ARC_SCALING_CASES, getArcScalingOperand} from './global-coordinate-scaling-fixtures';
import {expectArcScalingOutput} from './global-coordinate-output-assertions';

it('scaling inputs through 128 combined vertices match independent clipped counts and spherical area', () => {
  for (const reference of ARC_SCALING_CASES) {
    const operand = (right: boolean) => ({
      positions: Float32Array.from(getArcScalingOperand(reference.totalVertices, right).flat()),
      ringOffsets: Uint32Array.of(0, reference.totalVertices / 2),
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
    expect(Array.from(result.status)).toEqual([0]);
    expectArcScalingOutput(
      {
        positions: Array.from(result.positions),
        rings: Array.from(result.ringOffsets),
        counts: Array.from(result.counts),
        status: Array.from(result.status)
      },
      reference,
      false
    );
    expect(Array.from(result.counts)).toEqual([
      reference.intersectionVertices,
      1,
      reference.intersectionVertices,
      1
    ]);
    let area = 0;
    const count = result.counts[0];
    for (let row = 0; row < count; row++) {
      const first = Array.from(result.positions.subarray(row * 3, row * 3 + 3));
      const next = (row + 1) % count;
      const second = Array.from(result.positions.subarray(next * 3, next * 3 + 3));
      area +=
        2 *
        Math.atan2(
          first[0] * second[1] - first[1] * second[0],
          1 +
            first[2] +
            second[2] +
            first.reduce((sum, value, axis) => sum + value * second[axis], 0)
        );
    }
    expect(Number.isFinite(area)).toBe(true);
    expect(area).toBeCloseTo(reference.area, 6);
  }
});
