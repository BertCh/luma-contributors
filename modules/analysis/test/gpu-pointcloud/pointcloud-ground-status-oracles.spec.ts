// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUPointCloudGround} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';

it('ground status distinguishes full two-stage support, an edge and an unsupported point on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      new Float32Array([2.5, 2.5, 0, 0.5, 0.5, 0, 5, 0.5, 0])
    );
    const classifications = fixture.view('uint32', new Uint32Array([1, 1, 1]));
    const minimumSurface = fixture.view('float32', new Float32Array(25));
    const coverage = fixture.view('uint32', new Uint32Array(25).fill(1));
    const groundMask = fixture.view('uint32', new Uint32Array(3));
    const outputClassifications = fixture.view('uint32', new Uint32Array(3));
    const status = fixture.view('uint32', new Uint32Array(3).fill(0xffffffff));
    fixture.graph.add(
      new GPUPointCloudGround({
        positions: positions.view,
        classifications: classifications.view,
        grid: {origin: [0, 0], cellSize: 1, width: 5, height: 5},
        radii: [1],
        initialThreshold: 0,
        slope: 0,
        maximumThreshold: 0,
        minimumSurface: minimumSurface.view,
        coverage: coverage.view,
        output: {
          groundMask: groundMask.view,
          outputClassifications: outputClassifications.view,
          status: status.view
        }
      })
    );
    // Erosion then dilation with radius 1 requires a two-cell halo. Only
    // the center has that halo in this 5x5 raster. The corner is incomplete
    // (2), while x=5 is outside the half-open grid (1, excluding bit 2).
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0, 2, 1]);
    coverage.buffer.write(new Uint32Array(25));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([1, 1, 1]);
    coverage.buffer.write(new Uint32Array(25).fill(1));
    positions.buffer.write(new Float32Array([2.5, 2.5, 0, 2.5, 2.5, 0, 2.5, 2.5, 0]));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});
