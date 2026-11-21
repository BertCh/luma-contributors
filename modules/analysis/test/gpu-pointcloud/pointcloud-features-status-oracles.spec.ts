// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUPointCloudFeatures} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';

it('feature status uses diagonal-covariance stopping oracles and clears poisoned words on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // The six symmetric axis points have mean zero and population covariance
    // diag(1/3, 4/3, 3). Its off-diagonal residual is exactly zero before any
    // Jacobi rotation, so the not-converged bit must be zero independently
    // of an implementation's iteration count. The last row is collinear.
    const positions = fixture.view(
      'float32x3',
      new Float32Array([-1, 0, 0, 1, 0, 0, 0, -2, 0, 0, 2, 0, 0, 0, -3, 0, 0, 3])
    );
    const indices = fixture.view(
      'uint32',
      new Uint32Array([
        0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 0, 1, 0, 1
      ])
    );
    const counts = fixture.view('uint32', new Uint32Array([6, 2, 7, 0, 7]));
    const normals = fixture.view('float32x3', new Float32Array(15));
    const eigenvalues = fixture.view('float32x3', new Float32Array(15));
    const shapes = fixture.view('float32x3', new Float32Array(15));
    const status = fixture.view('uint32', new Uint32Array(5).fill(0xffffffff));
    fixture.graph.add(
      new GPUPointCloudFeatures({
        positions: positions.view,
        indices: indices.view,
        counts: counts.view,
        capacity: 6,
        output: {
          normals: normals.view,
          eigenvalues: eigenvalues.view,
          shapes: shapes.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    // Full-rank, insufficient, truncated, empty, and degenerate+truncated.
    expect(await readIntegers(status.buffer)).toEqual([0, 1, 2, 1, 3]);
    counts.buffer.write(new Uint32Array([0, 6, 0, 7, 6]));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([1, 0, 1, 2, 1]);
  } finally {
    fixture.destroy();
  }
});
