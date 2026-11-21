// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUPointCloudVoxelMerge} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';

it('voxel merge writes both overflow bits and clears their combinations on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const firstCounts = fixture.view('uint32', new Uint32Array([0xffffffff, 1, 0xffffffff]));
    const firstCentroids = fixture.view('float32x3', new Float32Array(9));
    const firstRepresentatives = fixture.view('uint32', new Uint32Array([0, 0, 0]));
    const secondCounts = fixture.view('uint32', new Uint32Array([1, 1, 1]));
    const secondCentroids = fixture.view('float32x3', new Float32Array(9));
    const secondRepresentatives = fixture.view(
      'uint32',
      new Uint32Array([0, 0xfffffffe, 0xfffffffe])
    );
    const counts = fixture.view('uint32', new Uint32Array(3));
    const centroids = fixture.view('float32x3', new Float32Array(9));
    const representatives = fixture.view('uint32', new Uint32Array(3));
    const status = fixture.view('uint32', new Uint32Array(3).fill(0xffffffff));
    fixture.graph.add(
      new GPUPointCloudVoxelMerge({
        partitionPolicy: 'disjoint-owned-rows',
        parts: [
          {
            counts: firstCounts.view,
            centroids: firstCentroids.view,
            representatives: firstRepresentatives.view,
            sourceRowOffset: 0
          },
          {
            counts: secondCounts.view,
            centroids: secondCentroids.view,
            representatives: secondRepresentatives.view,
            sourceRowOffset: 1
          }
        ],
        output: {
          counts: counts.view,
          centroids: centroids.view,
          representatives: representatives.view,
          status: status.view
        }
      })
    );
    // 0xffffffff + 1 = 2^32 exceeds uint32 (bit 1). The largest
    // publishable row is 0xfffffffe; 0xfffffffe + offset 1 is the
    // missing-row sentinel and cannot be published (bit 2).
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([1, 2, 3]);
    firstCounts.buffer.write(new Uint32Array([1, 1, 1]));
    secondRepresentatives.buffer.write(new Uint32Array([0, 1, 2]));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});
