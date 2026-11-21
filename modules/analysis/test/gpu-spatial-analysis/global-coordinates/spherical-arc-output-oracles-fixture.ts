// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {SphericalArcOverlayOperationProps} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay-operation';

// Three coordinate half-planes bound the positive octant. Intersecting it with itself
// preserves its oriented x -> y -> z ring: three quarter-circle edges and solid angle pi/2.
export const OCTANT_POSITIONS = [1, 0, 0, 0, 1, 0, 0, 0, 1];
export const OCTANT_PROPS: SphericalArcOverlayOperationProps = {
  operation: 'intersection',
  referencePoint: [0, 0, -1],
  capacity: 5,
  ringCapacity: 2,
  left: {positions: 'leftPositions', ringOffsets: 'leftOffsets', referenceInside: false},
  right: {positions: 'rightPositions', ringOffsets: 'rightOffsets', referenceInside: false},
  output: {positions: 'positions', ringOffsets: 'ringOffsets', counts: 'counts', status: 'status'}
};

export function expectOctant(result: Record<string, ArrayLike<number>>) {
  expect(Array.from(result['counts'])).toEqual([3, 1, 3, 0]);
  expect(Array.from(result['status'])).toEqual([0]);
  expect(Array.from(result['ringOffsets'])).toEqual([0, 3, 3]);
  const positions = Array.from(result['positions']);
  // Absolute 3e-6 covers float32 arc predicates; toBeLessThan rejects NaN.
  for (let index = 0; index < 9; index++)
    expect(Math.abs(positions[index] - OCTANT_POSITIONS[index])).toBeLessThan(3e-6);
  expect(positions.slice(9)).toEqual([0, 0, 0, 0, 0, 0]);
}
