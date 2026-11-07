// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {mapGeometryCoordinates} from '../src/analysis/catalog-geometry-utils';
import {solveDenseLinearSystem} from '../src/analysis/dense-linear-system';

test('solveDenseLinearSystem solves a small system in place', () => {
  const matrix = [new Float64Array([2, 1]), new Float64Array([1, 3])];
  const rightHandSide = [new Float64Array([5]), new Float64Array([10])];
  expect(solveDenseLinearSystem(matrix, rightHandSide)).toBe(true);
  expect(rightHandSide[0][0]).toBeCloseTo(1);
  expect(rightHandSide[1][0]).toBeCloseTo(3);
});

test('solveDenseLinearSystem scales its singularity tolerance with the matrix', () => {
  // Squared-meter magnitudes: a pivot of 1e-6 is relatively tiny, a pivot of 1e9 is not.
  expect(
    solveDenseLinearSystem(
      [
        [1e9, 0],
        [0, 1e-6]
      ],
      [[1], [1]]
    )
  ).toBe(false);
  expect(
    solveDenseLinearSystem(
      [
        [1e9, 0],
        [0, 1e9]
      ],
      [[1], [1]]
    )
  ).toBe(true);
  expect(
    solveDenseLinearSystem(
      [
        [1, 2],
        [2, 4]
      ],
      [[1], [2]]
    )
  ).toBe(false);
});

test('mapGeometryCoordinates preserves ordinates after the first two', () => {
  const mapped = mapGeometryCoordinates(
    [
      [1, 2, 30],
      [3, 4]
    ],
    ([x, y]) => [x + 1, y + 1]
  );
  expect(mapped).toEqual([
    [2, 3, 30],
    [4, 5]
  ]);
});
