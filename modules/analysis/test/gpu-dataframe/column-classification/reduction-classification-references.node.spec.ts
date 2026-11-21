// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeClassBreaksOracle, countClassesOracle} from './class-breaks-oracle';

it('DPC-110 independent oracle treats signed zeros as the same class value', () => {
  const customValues = Float32Array.of(-0, 0);
  expect(Array.from(countClassesOracle(customValues, undefined, [-1, 0, 1], 2, 2))).toEqual([0, 2]);
  const values = Float32Array.of(-1, -0, 0, 1);
  const result = computeClassBreaksOracle({
    values,
    parameters: {method: 'maximum-breaks', classCount: 4},
    maximumClassCount: 4,
    naturalBreaksBinCount: 16
  });
  expect(result.classCount).toBe(3);
  expect(Array.from(result.breaks).slice(0, 4)).toEqual([-1, -0.5, 0.5, 1]);
  expect(Number.isNaN(result.breaks[4])).toBe(true);
  expect(
    Array.from(countClassesOracle(values, undefined, result.breaks, result.classCount, 4))
  ).toEqual([1, 2, 1, 0]);
});

it('DPC-111 equal intervals have an exact symmetric midpoint without forming an f32 span', () => {
  const result = computeClassBreaksOracle({
    values: Float32Array.of(-(2 ** 127), 0, 2 ** 127),
    parameters: {method: 'equal-interval', classCount: 2},
    maximumClassCount: 2,
    naturalBreaksBinCount: 64
  });
  // Affine interpolation at 1/2 on [-a,a] is exactly zero.
  expect(Array.from(result.breaks)).toEqual([-(2 ** 127), 0, 2 ** 127]);
  expect(
    Array.from(
      countClassesOracle(Float32Array.of(-(2 ** 127), 0, 2 ** 127), undefined, result.breaks, 2, 2)
    )
  ).toEqual([1, 2]);
});

it.each([
  {
    values: Float32Array.of(2 ** -90, 2 ** -90, 2 ** -89, 2 ** 70),
    classes: 3,
    counts: [2, 1, 1],
    edges: [2 ** -90, 2 ** -89, 2 ** 70, 2 ** 70]
  },
  {
    values: Float32Array.of(-3e38, -2.9e38, -2.8e38, 2.8e38, 2.9e38, 3e38),
    classes: 2,
    counts: [3, 3],
    edges: Array.from(Float32Array.of(-3e38, 2.8e38, 3e38))
  }
])('DPC-111 natural mirror preserves input-unit edges and counts: $counts', ({
  values,
  classes,
  counts,
  edges
}) => {
  // Three distinct values and three classes have zero within-class SSD. For the symmetric
  // two-cluster fixture the sign split is the unique optimum (gap is 56x the cluster width).
  const result = computeClassBreaksOracle({
    values,
    parameters: {method: 'natural-breaks', classCount: classes},
    maximumClassCount: classes,
    naturalBreaksBinCount: 64
  });
  expect(result.classCount).toBe(classes);
  expect(Array.from(result.breaks)).toEqual(edges);
  expect(
    Array.from(countClassesOracle(values, undefined, result.breaks, classes, classes))
  ).toEqual(counts);
});

it('B2 integer-weight equal intervals preserve the exactly zero third-class cut', () => {
  const values = Float32Array.of(-2, -(2 ** -58), 0, 1);
  const result = computeClassBreaksOracle({
    values,
    parameters: {method: 'equal-interval', classCount: 3},
    maximumClassCount: 3,
    naturalBreaksBinCount: 64
  });
  // Width=(1-(-2))/3=1 exactly; the negative tiny row belongs below the zero cut.
  expect(Array.from(result.breaks)).toEqual([-2, -1, 0, 1]);
  expect(Array.from(countClassesOracle(values, undefined, result.breaks, 3, 3))).toEqual([1, 1, 2]);
});
