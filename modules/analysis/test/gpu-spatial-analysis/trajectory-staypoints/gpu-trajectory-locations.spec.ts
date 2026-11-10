// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  GPUTrajectoryLocations,
  getGPUTrajectoryLocationsParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphHarness} from './movement-spec-utils';

async function runLocations(
  centroids: number[][],
  tracks: number[],
  epsilon: number,
  maximumIterations = 32
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'locations-test');
  const rowCount = centroids.length;
  const locationIds = harness.output('location-ids', 'uint32', rowCount);
  const count = harness.output('count', 'uint32', 1);
  const converged = harness.output('converged', 'uint32', 1);
  const limit = harness.output('limit', 'uint32', 1);
  const centers = harness.output('centers', 'float32x2', rowCount);
  harness.graph.add(
    new GPUTrajectoryLocations({
      centroids: harness.input(
        'centroids',
        Float32Array.from(centroids.flat()),
        'float32x2',
        rowCount
      ),
      tracks: harness.input('tracks', Uint32Array.from(tracks), 'uint32', rowCount),
      count: harness.input('stay-count', Uint32Array.from([rowCount]), 'uint32', 1),
      parameters: harness.input(
        'parameters',
        getGPUTrajectoryLocationsParameterValues({epsilon}),
        'float32',
        4
      ),
      maximumIterations,
      output: {
        locationIds: locationIds.view,
        status: {count: count.view, converged: converged.view, iterationLimitReached: limit.view},
        centers: centers.view
      }
    })
  );
  return harness.run(async () => ({
    ids: await readUint32(locationIds.buffer, rowCount),
    count: (await readUint32(count.buffer, 1))[0],
    converged: (await readUint32(converged.buffer, 1))[0],
    iterationLimitReached: (await readUint32(limit.buffer, 1))[0],
    centers: await readFloat32(centers.buffer, rowCount * 2)
  }));
}

describe('GPUTrajectoryLocations', () => {
  it('connects staypoints transitively with an inclusive epsilon, per track', async () => {
    // Expected labels computed from the definition by hand: rows 0, 1, 2 chain at spacing 100.
    const centroids = [
      [0, 0],
      [100, 0],
      [200, 0],
      [1000, 0],
      [1000, 40],
      [0, 0]
    ];
    const tracks = [0, 0, 0, 0, 0, 1];
    const inclusive = await runLocations(centroids, tracks, 100);
    expect(inclusive.ids).toEqual([0, 0, 0, 1, 1, 2]);
    expect(inclusive.count).toBe(3);
    expect(inclusive.converged).toBe(1);
    expect(inclusive.iterationLimitReached).toBe(0);
    expect(inclusive.centers.slice(0, 2)).toEqual([100, 0]); // distinct mean of 0, 100, 200
    const tighter = await runLocations(centroids, tracks, 99);
    expect(tighter.ids).toEqual([0, 1, 2, 3, 3, 4]);
    expect(tighter.count).toBe(5);
  });

  it('reports a chain that needs more rounds than allowed', async () => {
    // A long chain, labelled by hooking and path halving, cannot resolve in one round.
    const centroids = Array.from({length: 64}, (_, index) => [index * 50, 0]);
    const tracks = new Array(64).fill(0);
    const result = await runLocations(centroids, tracks, 50, 1);
    expect(result.converged).toBe(0);
    expect(result.iterationLimitReached).toBe(1);
  });
});
