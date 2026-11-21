// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {prepareSamplingTriangleFrame} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/points/sampling-triangle-frame';
import {triangulateMonotoneOnCPU} from '../../../analysis/src/analysis-core/polygon-triangulation/cpu-monotone-triangulation';
it('GEO-124 filters two-touching-hole zero-area rows and retains positive row provenance', () => {
  // Hand area: 10*10 - 2*2 - 2*2 = 92; holes touch only at (4,4).
  const rings: [number, number][][] = [
    [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10]
    ],
    [
      [2, 2],
      [2, 4],
      [4, 4],
      [4, 2]
    ],
    [
      [4, 4],
      [4, 6],
      [6, 6],
      [6, 4]
    ]
  ];
  const triangles = triangulateMonotoneOnCPU(rings).triangles;
  expect(triangles).toHaveLength(14);
  const result = prepareSamplingTriangleFrame(
    Float32Array.from(rings.flat(2)),
    Uint32Array.from(triangles.flat()),
    triangles.length
  );
  expect(Array.from(result.sourceTriangleRows)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13]);
  expect(result.positions.length).toBe(6 * result.sourceTriangleRows.length);
  let area = 0;
  for (let row = 0; row < result.sourceTriangleRows.length; row++) {
    const offset = row * 6;
    const values = result.positions;
    const determinant =
      (values[offset + 2] - values[offset]) * (values[offset + 5] - values[offset + 1]) -
      (values[offset + 3] - values[offset + 1]) * (values[offset + 4] - values[offset]);
    expect(Number.isFinite(determinant)).toBe(true);
    expect(Math.abs(determinant)).toBeGreaterThan(0);
    area += Math.abs(determinant) / 2;
    expect(Array.from(values.subarray(offset, offset + 6))).toEqual(
      triangles[result.sourceTriangleRows[row]].flatMap(vertex => rings.flat()[vertex])
    );
  }
  expect(area).toBe(92);
  expect(
    prepareSamplingTriangleFrame(new Float32Array([0, 0]), new Uint32Array([0, 0, 0]), 1).positions
  ).toHaveLength(0);
});
it('B5 retains numerically usable corner ordering beside a good row', () => {
  const result = prepareSamplingTriangleFrame(
    new Float32Array([8192, 8192, -8192, -8192, -8192, -8191.99951171875, 0, 0, 1, 0, 0, 1]),
    new Uint32Array([0, 1, 2, 3, 4, 5]),
    2
  );
  expect(Array.from(result.sourceTriangleRows)).toEqual([0, 1]);
  expect(result.positions).toHaveLength(12);
  for (let offset = 0; offset < 12; offset += 6) {
    const values = result.positions;
    const x = Math.fround(values[offset + 2] - values[offset]);
    const y = Math.fround(values[offset + 3] - values[offset + 1]);
    const nextX = Math.fround(values[offset + 4] - values[offset]);
    const nextY = Math.fround(values[offset + 5] - values[offset + 1]);
    expect(Math.abs(Math.fround(Math.fround(x * nextY) - Math.fround(y * nextX)))).toBeGreaterThan(
      0
    );
  }
});
import {getSamplingSourceRow} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/points/sampling-triangle-frame';
it('B6 preserves legacy row IDs while remapping random area rows', () => {
  expect(getSamplingSourceRow(0, 2, new Uint32Array([1]))).toBe(0);
  expect(getSamplingSourceRow(0, 0, new Uint32Array([1]))).toBe(1);
});
it('B5 explicitly refuses nonzero triangles with no usable float32 corner', () => {
  expect(() =>
    prepareSamplingTriangleFrame(
      new Float32Array([0, 0, 2 ** -149, 0, 0, 2 ** -149]),
      new Uint32Array([0, 1, 2]),
      1
    )
  ).toThrow(/float32/);
});
