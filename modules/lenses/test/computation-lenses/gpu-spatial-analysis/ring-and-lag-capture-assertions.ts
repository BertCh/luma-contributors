// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {LensRecord} from '../../../src/computation-views/registry/view-registration';
import {
  getCaptureFirstKey,
  getCaptureValue
} from '../../../src/computation-views/views/record-queries';

function capture(record: LensRecord, name: string) {
  const matches = record.captures!.snapshots.filter(snapshot => snapshot.name === name);
  expect(matches, name).toHaveLength(1);
  return matches[0];
}

function area(positions: ArrayLike<number>, first: number, last: number) {
  const originX = Number(positions[2 * first]);
  const originY = Number(positions[2 * first + 1]);
  let twiceArea = 0;
  for (let vertex = first; vertex < last - 1; vertex++) {
    twiceArea +=
      (Number(positions[2 * vertex]) - originX) * (Number(positions[2 * vertex + 3]) - originY) -
      (Number(positions[2 * vertex + 2]) - originX) * (Number(positions[2 * vertex + 1]) - originY);
  }
  return twiceArea / 2;
}

export function assertRingCapture(record: LensRecord) {
  expect(Array.from(capture(record, 'overflow').values)).toEqual([0]);
  expect(Array.from(capture(record, 'openSegmentCount').values)).toEqual([0]);
  // Shapely union of the float32 input is one polygon with no holes, not 17 uncancelled shells.
  expect(Array.from(capture(record, 'ringCount').values)).toEqual([1]);
  expect(Number(capture(record, 'ringIsHole').values[0])).toBe(0);
  const count = Number(capture(record, 'ringCount').values[0]);
  expect(Number.isInteger(count) && count > 0).toBe(true);
  const offsets = capture(record, 'ringOffsets').values;
  const positions = capture(record, 'ringPositions').values;
  const reportedAreas = capture(record, 'ringAreas').values;
  expect(count).toBeLessThan(offsets.length);
  let outputArea = 0;
  let outputPerimeter = 0;
  for (let ring = 0; ring < count; ring++) {
    const first = Number(offsets[ring]);
    const last = Number(offsets[ring + 1]);
    expect(Number.isInteger(first) && Number.isInteger(last)).toBe(true);
    expect(first).toBeGreaterThanOrEqual(0);
    expect(last - first).toBeGreaterThanOrEqual(4);
    expect(last * 2).toBeLessThanOrEqual(positions.length);
    expect(Array.from(positions).slice(2 * first, 2 * first + 2)).toEqual(
      Array.from(positions).slice(2 * last - 2, 2 * last)
    );
    const measured = area(positions, first, last);
    expect(Number.isFinite(measured) && Number.isFinite(Number(reportedAreas[ring]))).toBe(true);
    expect(Math.abs(Number(reportedAreas[ring]) - measured)).toBeLessThan(100);
    outputArea += measured;
    for (let vertex = first; vertex < last - 1; vertex++) {
      outputPerimeter += Math.hypot(
        Number(positions[2 * vertex + 2]) - Number(positions[2 * vertex]),
        Number(positions[2 * vertex + 3]) - Number(positions[2 * vertex + 1])
      );
    }
  }
  // Shapely unary_union of the 17 input polygons after float32 upload: 119839160.422753 m².
  // The polygons have disjoint interiors; their shared borders cancel, so the sum also agrees.
  // Independent reference and reproduction are in ring-and-lag-capture-oracles.oracle.py.
  const inputPositions = capture(record, 'positions').values;
  const inputOffsets = capture(record, 'areaRingOffsets').values;
  let inputArea = 0;
  for (let ring = 0; ring < inputOffsets.length - 1; ring++)
    inputArea += Math.abs(
      area(inputPositions, Number(inputOffsets[ring]), Number(inputOffsets[ring + 1]))
    );
  expect(Number.isFinite(inputArea) && Number.isFinite(outputArea)).toBe(true);
  expect(Math.abs(inputArea - 119839160.422753)).toBeLessThan(1);
  expect(Math.abs(outputArea - 119839160.422753)).toBeLessThan(1200);
  // Independent Shapely union outline length; the sum of all input perimeters is 220150.90018182.
  expect(Number.isFinite(outputPerimeter)).toBe(true);
  expect(Math.abs(outputPerimeter - 80373.20290219603)).toBeLessThan(0.1);
}

export function assertLagMLCapture(record: LensRecord) {
  for (const name of ['status', 'weightsOverflow', 'weightsCandidateOverflow'])
    expect(Array.from(capture(record, name).values), name).toEqual([0]);
  const estimates = capture(record, 'estimates');
  expect(estimates.keySpace).toBe('coefficient');
  expect(estimates.components ?? 1).toBe(1);
  expect(estimates.values).toHaveLength(5);
  expect(getCaptureFirstKey(estimates)).toBe(0);
  // Independent float64 NumPy least squares + SciPy concentrated likelihood, exact queen W,
  // row standardized; trace series Q=128. See the sibling Python reference.
  const expected = [0.78834312, 0.14601426, -0.01152027, -0.00772924, 0.6499915484];
  for (const [row, reference] of expected.entries()) {
    const actual = Number(getCaptureValue(estimates, row));
    expect(Number.isFinite(actual), `coefficient row ${row}`).toBe(true);
    expect(Math.abs(actual - reference), `estimate row ${row}`).toBeLessThan(0.002);
  }
  expect(Math.abs(Number(getCaptureValue(estimates, 4)) - 0.6499915484)).toBeLessThan(0.002);
}
