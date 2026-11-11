// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeMinimumBoundsOnCPU,
  CPUMinimumBoundsOperation
} from '../../../src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation';
import {
  GPUMinimumBoundsOperation,
  type MinimumBoundsOperationProps
} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation';
import fixtures from './minimum-bounds-fixture.json';

it('matches GEOS circles/rectangles/hulls and SciPy all-point diameters on 24 larger shapes', () => {
  for (const fixture of fixtures) {
    const result = computeMinimumBoundsOnCPU({
      positions: new Float32Array(fixture.positions),
      labels: new Uint32Array(fixture.positions.length / 2),
      groupCount: 1
    });
    expect(result.hullSizes[0]).toBe(fixture.hullSize);
    expect(Math.abs(result.rectangleSizes[3] - fixture.rectangleArea)).toBeLessThanOrEqual(
      2e-6 * fixture.rectangleArea
    );
    expect(result.diameters[0]).toBe(Math.fround(fixture.diameter));
    expect(Array.from(result.bounds)).toEqual(fixture.bounds);
    for (let component = 0; component < 3; component++)
      expect(Math.abs(result.circles[component] - fixture.circle[component])).toBeLessThanOrEqual(
        2e-6 * Math.max(1, Math.abs(fixture.circle[component]))
      );
  }
});

it('matches hand-computed right-triangle, segment, point and empty-group shapes', () => {
  // A 3-4-5 right triangle has circumcircle (2, 1.5), radius 2.5; its hypotenuse
  // is the diameter. The first rectangle edge wins the three equal-area rectangles.
  const result = computeMinimumBoundsOnCPU({
    positions: new Float32Array([0, 0, 4, 0, 0, 3, 1, 1, 10, 2, 13, 6, 13, 6, -2, 7]),
    labels: new Uint32Array([0, 0, 0, 0, 1, 1, 1, 2]),
    groupCount: 4
  });
  expect(Array.from(result.hullSizes)).toEqual([3, 2, 1, 0]);
  expect(Array.from(result.rectangleCorners.slice(0, 8))).toEqual([0, 0, 4, 0, 4, 3, 0, 3]);
  expect(Array.from(result.rectangleSizes.slice(0, 4))).toEqual([4, 3, 0, 12]);
  // A 3-4-5 segment's rectangle is the segment traversed forward and back, with no thickness.
  expect(Array.from(result.rectangleSizes.slice(4, 8))).toEqual([
    5,
    0,
    Math.fround(Math.atan2(4, 3)),
    0
  ]);
  expect(Array.from(result.rectangleCorners.slice(8, 16))).toEqual([10, 2, 13, 6, 13, 6, 10, 2]);
  expect(Array.from(result.circles.slice(0, 9))).toEqual([2, 1.5, 2.5, 11.5, 4, 2.5, -2, 7, 0]);
  expect(Array.from(result.longestLines.slice(0, 12))).toEqual([
    4, 0, 0, 3, 10, 2, 13, 6, -2, 7, -2, 7
  ]);
  expect(Array.from(result.diameters.slice(0, 3))).toEqual([5, 5, 0]);
  expect(Array.from(result.bounds.slice(0, 12))).toEqual([0, 0, 4, 3, 10, 2, 13, 6, -2, 7, -2, 7]);
  for (const [port, width] of [
    ['rectangleCorners', 8],
    ['rectangleSizes', 4],
    ['circles', 3],
    ['longestLines', 4],
    ['diameters', 1],
    ['bounds', 4]
  ] as const) {
    expect(result[port].slice(3 * width).every(Number.isNaN)).toBe(true);
  }
  expect(Array.from(result.overflow)).toEqual([0]);
});

it('uses a rotated square closed form and exact lowest-hull-pair diameter ties', () => {
  const result = computeMinimumBoundsOnCPU({
    positions: new Float32Array([10, 0, 12, 1, 11, 3, 9, 2, 10, 1]),
    labels: new Uint32Array(5),
    groupCount: 1
  });
  expect(result.rectangleSizes[0]).toBeCloseTo(Math.sqrt(5), 6);
  expect(result.rectangleSizes[1]).toBeCloseTo(Math.sqrt(5), 6);
  expect(result.rectangleSizes[3]).toBe(5);
  expect(Array.from(result.rectangleCorners)).toEqual([9, 2, 10, 0, 12, 1, 11, 3]);
  expect(Array.from(result.longestLines)).toEqual([9, 2, 12, 1]);
  expect(result.diameters[0]).toBe(Math.fround(Math.sqrt(10)));
  expect(Array.from(result.circles)).toEqual([10.5, 1.5, Math.fround(Math.sqrt(2.5))]);
});

it('derives feature labels through empty polygon/ring intervals and excludes uncovered vertices', () => {
  const result = computeMinimumBoundsOnCPU({
    positions: new Float32Array([99, 99, 0, 0, 4, 0, 0, 3, 100, 100]),
    ringOffsets: new Uint32Array([1, 1, 4]),
    polygonOffsets: new Uint32Array([0, 0, 2]),
    featureOffsets: new Uint32Array([0, 1, 2])
  });
  expect(Array.from(result.hullSizes)).toEqual([0, 3]);
  expect(Array.from(result.bounds.slice(4))).toEqual([0, 0, 4, 3]);
});

it('excludes nonfinite positions and unassigned labels and rejects inconsistent arrays', () => {
  expect(
    computeMinimumBoundsOnCPU({
      positions: new Float32Array([NaN, 0, 1, 2, 5, 5]),
      labels: new Uint32Array([0, 0, 0xffffffff]),
      groupCount: 1
    }).bounds
  ).toEqual(new Float32Array([1, 2, 1, 2]));
  expect(() =>
    computeMinimumBoundsOnCPU({
      positions: new Float32Array(3),
      labels: new Uint32Array(1),
      groupCount: 1
    })
  ).toThrow(/positions/);
  expect(() =>
    computeMinimumBoundsOnCPU({
      positions: new Float32Array(4),
      labels: new Uint32Array(1),
      groupCount: 1
    })
  ).toThrow(/labels/);
  for (const ringOffsets of [
    new Uint32Array([0]),
    new Uint32Array([2, 1]),
    new Uint32Array([0, 3])
  ]) {
    expect(() =>
      computeMinimumBoundsOnCPU({
        positions: new Float32Array(4),
        ringOffsets,
        polygonOffsets: new Uint32Array([0, 1]),
        featureOffsets: new Uint32Array([0, 1])
      })
    ).toThrow(/ringOffsets/);
  }
});

it('validates operation output contracts at construction and CPU numeric props', () => {
  const props: MinimumBoundsOperationProps = {
    id: 'bounds-check',
    positions: 'positions',
    labels: 'labels',
    rowCount: 3,
    groupCount: 1,
    output: {bounds: 'bounds', status: {overflow: 'overflow'}}
  };
  for (const Operation of [CPUMinimumBoundsOperation, GPUMinimumBoundsOperation]) {
    expect(() => new Operation(props)).not.toThrow();
    expect(() => new Operation({...props, groupCount: 0})).toThrow(/^bounds-check/);
    expect(
      () => new Operation({...props, output: {bounds: 'positions', status: {overflow: 'overflow'}}})
    ).toThrow(/^bounds-check/);
  }
  expect(() => new CPUMinimumBoundsOperation({...props, rowCount: 1.5})).toThrow(
    /^bounds-check.*rowCount/
  );
  expect(() => new CPUMinimumBoundsOperation({...props, maximumHullVertices: 2})).toThrow(
    /^bounds-check.*maximumHullVertices/
  );
});
