// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {MinimumBoundsOperationProps} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation';

// Right triangle (0,0),(4,0),(0,3): the three edge-aligned bounding rectangles
// all have area 12. The documented first-edge tie break chooses the horizontal edge.
// The hypotenuse endpoints are the unique farthest pair, distance 5. Its midpoint
// (2,1.5) and radius 2.5 enclose the right triangle (Thales). The fourth point is
// strictly inside; the second group is a singleton at (9,7), exercising the last row.
export const BOUNDS_POSITIONS = [0, 0, 4, 0, 0, 3, 1, 1, 9, 7];
export const BOUNDS_LABELS = [0, 0, 0, 0, 1];
export const BOUNDS_EXPECTED = {
  rectangleCorners: [0, 0, 4, 0, 4, 3, 0, 3, 9, 7, 9, 7, 9, 7, 9, 7],
  rectangleSizes: [4, 3, 0, 12, 0, 0, 0, 0],
  circles: [2, 1.5, 2.5, 9, 7, 0],
  longestLines: [4, 0, 0, 3, 9, 7, 9, 7],
  diameters: [5, 0],
  sourceBounds: [0, 0, 4, 3, 9, 7, 9, 7],
  hullSizes: [3, 1],
  overflow: [0],
  groupOverflow: [0]
};

const BOUNDS_SHAPE_COLUMNS = [
  'rectangleCorners',
  'rectangleSizes',
  'circles',
  'longestLines',
  'diameters',
  'sourceBounds'
] as const;

// hullSizes and groupOverflow are metadata, so each requires one actual shape output.
// bounds is their sole shape dependency; its independently pinned values must remain unchanged.
export const BOUNDS_OUTPUT_CASES: readonly (readonly (keyof typeof BOUNDS_EXPECTED)[])[] = [
  ...BOUNDS_SHAPE_COLUMNS.map(column => [column]),
  ['sourceBounds', 'hullSizes'],
  [...BOUNDS_SHAPE_COLUMNS, 'hullSizes'],
  ['sourceBounds', 'groupOverflow']
];

export function getBoundsOracleOperationProps(
  selected: readonly (keyof typeof BOUNDS_EXPECTED)[]
): MinimumBoundsOperationProps {
  const output: MinimumBoundsOperationProps['output'] = {status: {overflow: 'overflow'}};
  for (const port of selected) {
    if (port === 'groupOverflow') output.status.groupOverflow = 'groupOverflow';
    else if (port !== 'overflow') Object.assign(output, {[port]: port});
  }
  return {positions: 'positions', labels: 'labels', rowCount: 5, groupCount: 2, output};
}

export function expectBoundsOutputPorts(
  selected: readonly (keyof typeof BOUNDS_EXPECTED)[],
  outputs: Readonly<Record<string, unknown>>
) {
  expect(Object.keys(outputs).sort()).toEqual(['overflow', ...selected].sort());
}

export function expectBoundsPort(port: keyof typeof BOUNDS_EXPECTED, actual: ArrayLike<number>) {
  const values = Array.from(actual);
  const expected = BOUNDS_EXPECTED[port];
  expect(values.length, port).toBe(expected.length);
  if (port === 'hullSizes' || port === 'overflow' || port === 'groupOverflow')
    expect(values, port).toEqual(expected);
  // 3e-5 absolute is conservative for float32 unit-vector projections on this <=12 scale.
  else
    for (let index = 0; index < expected.length; index++)
      expect(Math.abs(values[index] - expected[index]), `${port}[${index}]`).toBeLessThan(3e-5);
}
