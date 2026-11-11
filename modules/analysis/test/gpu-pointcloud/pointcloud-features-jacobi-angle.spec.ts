// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// FX-17: the Jacobi rotation of `GPUPointCloudFeatures` uses
// `0.5 * atan2(2 * a_pq, a_qq - a_pp)`. Neighbourhoods with equal diagonal covariance entries make
// the second argument exactly zero. A difference of equal values is +0.0, and -0.0 needs
// `a_qq = -0.0` with `a_pp = +0.0`, which a positive semidefinite covariance with a nonzero
// `a_pq` cannot have (the pivot is skipped when `a_pq` is below 1e-30). Either sign of the
// quarter turn also diagonalizes the pair, so the site is not affected; this spec pins it against
// a closed-form binary64 reference.

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUPointCloudFeatures} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

/** Rows of points whose covariance has equal x and y variances and a nonzero xy covariance. */
const NEIGHBORHOODS: number[][][] = [
  [
    [0, 0, 0],
    [1, 1, 0],
    [-1, -1, 0],
    [2, -2, 0],
    [-2, 2, 0]
  ],
  [
    [0, 0, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [3, 3, 0],
    [-3, -3, 0]
  ],
  [
    [0, 0, 0],
    [0, 1, 1],
    [0, -1, -1],
    [0, 2, -2],
    [0, -2, 2]
  ]
];
const CAPACITY = 5;

/** Closed-form eigenvalues of the population covariance with equal variances in one plane. */
function getReference(points: number[][]): {values: number[]; normal: number[]} {
  const count = points.length;
  const mean = [0, 1, 2].map(axis => points.reduce((sum, point) => sum + point[axis], 0) / count);
  const covariance = [0, 1, 2].map(row =>
    [0, 1, 2].map(
      column =>
        points.reduce(
          (sum, point) => sum + (point[row] - mean[row]) * (point[column] - mean[column]),
          0
        ) / count
    )
  );
  const flatAxis = [0, 1, 2].find(axis => covariance[axis][axis] === 0)!;
  const [first, second] = [0, 1, 2].filter(axis => axis !== flatAxis);
  const variance = covariance[first][first];
  const offDiagonal = Math.abs(covariance[first][second]);
  const normal = [0, 0, 0];
  normal[flatAxis] = 1;
  return {values: [variance + offDiagonal, variance - offDiagonal, 0], normal};
}

it('GPUPointCloudFeatures matches binary64 where the Jacobi atan2 second argument is zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  const rowCount = NEIGHBORHOODS.length;
  try {
    const positions = fixture.view('float32x3', new Float32Array(NEIGHBORHOODS.flat(2)));
    const indices = fixture.view(
      'uint32',
      Uint32Array.from({length: rowCount * CAPACITY}, (_, index) => index)
    );
    const counts = fixture.view('uint32', new Uint32Array(rowCount).fill(CAPACITY));
    const normals = fixture.view('float32x3', new Float32Array(rowCount * 3));
    const eigenvalues = fixture.view('float32x3', new Float32Array(rowCount * 3));
    const shapes = fixture.view('float32x3', new Float32Array(rowCount * 3));
    const status = fixture.view('uint32', new Uint32Array(rowCount));
    fixture.graph.add(
      new GPUPointCloudFeatures({
        positions: positions.view,
        indices: indices.view,
        counts: counts.view,
        normals: normals.view,
        eigenvalues: eigenvalues.view,
        shapes: shapes.view,
        status: status.view,
        capacity: CAPACITY
      })
    );
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual(Array(rowCount).fill(0));
    const actualNormals = await readFloats(normals.buffer);
    const actualValues = await readFloats(eigenvalues.buffer);
    NEIGHBORHOODS.forEach((points, row) => {
      const reference = getReference(points);
      const values = actualValues.slice(row * 3, row * 3 + 3);
      const sortedValues = [...values].sort((left, right) => right - left);
      reference.values.forEach((value, index) =>
        expect(sortedValues[index], `row ${row} eigenvalue ${index}`).toBeCloseTo(value, 5)
      );
      reference.normal.forEach((value, axis) =>
        expect(Math.abs(actualNormals[row * 3 + axis]), `row ${row} normal ${axis}`).toBeCloseTo(
          value,
          5
        )
      );
    });
  } finally {
    fixture.destroy();
  }
});
