// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  createGroundSeedCoverageNode,
  GPUPointCloudNeighborhoods,
  GPUPointCloudNormalScale
} from '@luma-contributors/analysis/pointcloud-analysis';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';

it('UNI-01: single-output public seed conversion normalizes only ground coverage', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const coverage = fixture.view('uint32', Uint32Array.of(0, 1, 2, 3, 4, 6, 7));
    const output = fixture.view('uint32', new Uint32Array(7).fill(99));
    fixture.graph.add(createGroundSeedCoverageNode(fixture.graph, coverage.view, output.view));
    await fixture.run();
    expect(await readIntegers(output.buffer)).toEqual([0, 0, 1, 1, 0, 1, 1]);
  } finally {
    fixture.destroy();
  }
});

it('BC3: smaller-scale support excludes a rounded-key boundary neighbor and selects radius two', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // Binary32 coordinates: first squared norm = 1.0000000099999995 > 1,
    // although its binary32 key is 1. Radius-one support is 2, radius-two support 3.
    // Three noncollinear points lie exactly in z=0, hence normal (0,0,1), variance 0.
    const positions = fixture.view(
      'float32x3',
      Float32Array.of(1, 0.0001, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0)
    ).view;
    const queryRows = fixture.view('uint32', Uint32Array.of(3)).view;
    const counts = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudNeighborhoods({
        positions,
        queryRows,
        capacity: 3,
        radius: 1,
        output: {
          indices: fixture.view('uint32', new Uint32Array(3)).view,
          squaredDistances: fixture.view('float32', new Float32Array(3)).view,
          counts: counts.view
        }
      })
    );
    const normals = fixture.view('float32x3', new Float32Array(3));
    const normalScales = fixture.view('float32', new Float32Array(1));
    const normalVariances = fixture.view('float32', new Float32Array(1));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudNormalScale({
        positions,
        queryRows,
        capacity: 3,
        radii: [1, 2],
        parameters: fixture.view('float32', Float32Array.of(3, 0, 0, 0, 0)).view,
        output: {
          normals: normals.view,
          normalScales: normalScales.view,
          normalVariances: normalVariances.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    expect(await readIntegers(counts.buffer)).toEqual([2]);
    expect(await readIntegers(status.buffer)).toEqual([0]);
    expect(await readFloats(normalScales.buffer)).toEqual([2]);
    expect(await readFloats(normalVariances.buffer)).toEqual([0]);
    expect(await readFloats(normals.buffer)).toEqual([0, 0, 1]);
  } finally {
    fixture.destroy();
  }
});
