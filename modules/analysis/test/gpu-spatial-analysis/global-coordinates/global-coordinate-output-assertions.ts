// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect} from 'vitest';
import type {ARC_SCALING_CASES} from './global-coordinate-scaling-fixtures';

type ArcReference = (typeof ARC_SCALING_CASES)[number];
export function expectArcScalingOutput(
  output: Record<string, number[]>,
  reference: ArcReference,
  updated: boolean
) {
  expect(output['status']).toEqual([0]);
  expect(output['counts']).toEqual([
    reference.intersectionVertices,
    1,
    reference.intersectionVertices,
    1
  ]);
  expect(output['positions']).toHaveLength(768);
  const expectedVertex = updated ? reference.rotatedBoundaryVertex : reference.boundaryVertex;
  expect(
    output['positions'].slice(0, reference.intersectionVertices * 3).every(Number.isFinite)
  ).toBe(true);
  expect(
    Array.from({length: reference.intersectionVertices}, (_, row) =>
      Math.hypot(
        ...expectedVertex.map((value, axis) => output['positions'][row * 3 + axis] - value)
      )
    ).some(distance => distance < 2e-5)
  ).toBe(true);
  let area = 0;
  for (let ring = 0; ring < output['counts'][1]; ring++)
    for (let row = output['rings'][ring]; row < output['rings'][ring + 1]; row++) {
      const first = output['positions'].slice(row * 3, row * 3 + 3);
      const next = row + 1 === output['rings'][ring + 1] ? output['rings'][ring] : row + 1;
      const second = output['positions'].slice(next * 3, next * 3 + 3);
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
  if (area < 0) area += 4 * Math.PI;
  expect(Number.isFinite(area)).toBe(true);
  expect(Math.abs(area - reference.area)).toBeLessThan(2e-5);
}
export function expectDatumSelectionPositions(
  positions: number[],
  reverse: boolean,
  nodata: boolean
) {
  expect(positions).toEqual(
    nodata
      ? [NaN, NaN, NaN, NaN, NaN, NaN]
      : reverse
        ? [0, 1, 10, NaN, NaN, NaN]
        : [0.75, 0.375, 10, NaN, NaN, NaN]
  );
}
export function expectDatumGridCountPositions(
  positions: number[],
  gridCount: number,
  updated: boolean
) {
  if (updated)
    expect(positions).toEqual([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]);
  else
    expect(positions).toEqual(
      Array(4)
        .fill([gridCount + 9.75, 0.375, 10])
        .flat()
    );
}
export function expectConvexReadbackLengths(positions: number[], measures: number[]) {
  expect(positions).toHaveLength(48);
  expect(measures).toHaveLength(8);
  expect(positions.every(Number.isFinite)).toBe(true);
  expect(measures.every(Number.isFinite)).toBe(true);
}
export function getDatumSourceObservation(source: string, gridCount: number) {
  return {gridCount, generatedBytes: new TextEncoder().encode(source).length};
}
