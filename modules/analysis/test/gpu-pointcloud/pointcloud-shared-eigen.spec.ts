// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudFeatures,
  GPUPointCloudICP
} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

for (const scale of [1e-6, 1, 1e6]) {
  it(`feature eigenpairs preserve a known rotated covariance at scale ${scale}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    // Three orthonormal axes, with lengths 3, 2, 1. Symmetric pairs have zero centroid,
    // so population eigenvalues are exactly length squared / 3, independent of a solver.
    const first = [1 / Math.sqrt(5), 2 / Math.sqrt(5), 0];
    const normal = [2 / 3, -1 / 3, 2 / 3];
    const second = [
      normal[1] * first[2] - normal[2] * first[1],
      normal[2] * first[0] - normal[0] * first[2],
      normal[0] * first[1] - normal[1] * first[0]
    ];
    const positions = [first, second, normal].flatMap((axis, index) =>
      [1, -1].flatMap(sign => axis.map(value => value * (3 - index) * sign * scale))
    );
    try {
      const normals = fixture.view('float32x3', new Float32Array(3));
      const eigenvalues = fixture.view('float32x3', new Float32Array(3));
      const status = fixture.view('uint32', new Uint32Array(1));
      fixture.graph.add(
        new GPUPointCloudFeatures({
          positions: fixture.view('float32x3', new Float32Array(positions)).view,
          indices: fixture.view('uint32', new Uint32Array([0, 1, 2, 3, 4, 5])).view,
          counts: fixture.view('uint32', new Uint32Array([6])).view,
          capacity: 6,
          output: {
            normals: normals.view,
            eigenvalues: eigenvalues.view,
            shapes: fixture.view('float32x3', new Float32Array(3)).view,
            status: status.view
          }
        })
      );
      await fixture.run();
      expect(await readIntegers(status.buffer)).toEqual([0]);
      const actualNormals = await readFloats(normals.buffer);
      normal.forEach((value, axis) => expect(actualNormals[axis]).toBeCloseTo(value, 5));
      const actualValues = await readFloats(eigenvalues.buffer);
      [3, 4 / 3, 1 / 3].forEach((value, axis) =>
        expect(actualValues[axis] / (scale * scale)).toBeCloseTo(value, 5)
      );
    } finally {
      fixture.destroy();
    }
  });

  it(`Horn eigenpairs recover a known rigid transform at scale ${scale}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    const source = [
      [-1, -1, -1],
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 3],
      [2, -2, 1]
    ].map(point => point.map(value => value * scale));
    const cosine = Math.cos(0.03);
    const sine = Math.sin(0.03);
    const translation = [0.02, -0.03, 0.01].map(value => value * scale);
    const target = source.map(point => [
      cosine * point[0] - sine * point[1] + translation[0],
      sine * point[0] + cosine * point[1] + translation[1],
      point[2] + translation[2]
    ]);
    try {
      const transform = fixture.view('float32', new Float32Array(16));
      const status = fixture.view('uint32', new Uint32Array(1));
      const inlierMask = fixture.view('uint32', new Uint32Array(5));
      fixture.graph.add(
        new GPUPointCloudICP({
          source: fixture.view('float32x3', new Float32Array(source.flat())).view,
          target: fixture.view('float32x3', new Float32Array(target.flat())).view,
          maximumDistance: 0.2 * scale,
          maximumIterations: 6,
          translationTolerance: 1e-5 * scale,
          rotationTolerance: 1e-5,
          output: {
            transform: transform.view,
            status: status.view,
            inlierMask: inlierMask.view,
            history: fixture.view('float32x3', new Float32Array(18)).view
          }
        })
      );
      await fixture.run();
      expect(await readIntegers(status.buffer)).toEqual([0]);
      expect(await readIntegers(inlierMask.buffer)).toEqual([1, 1, 1, 1, 1]);
      const actual = await readFloats(transform.buffer);
      const expected = [cosine, sine, 0, 0, -sine, cosine, 0, 0, 0, 0, 1, 0, ...translation, 1];
      expected.forEach((value, index) =>
        expect(Math.abs(actual[index] - value)).toBeLessThan(
          3e-5 * (index >= 12 && index <= 14 ? scale : 1)
        )
      );
    } finally {
      fixture.destroy();
    }
  });
}
