// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUPointCloudNeighborhoods} from '@luma-contributors/graph/gpu-pointcloud';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

it('sorted 3D grid neighborhoods equal all-pairs across cell boundaries, duplicates and repeat encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const values = Array.from({length: 512}, (_, index) => [
      (index % 8) + 0.125,
      (Math.floor(index / 8) % 8) + 0.125,
      Math.floor(index / 64) + 0.125
    ]).flat();
    values.push(0.125, 0.125, 0.125, -100, 0, 0);
    const positions = fixture.view('float32x3', new Float32Array(values));
    const count = values.length / 3;
    const results = [false, true].map(indexed => {
      const indices = fixture.view('uint32', new Uint32Array(count * 4));
      const distances = fixture.view('float32', new Float32Array(count * 4));
      const counts = fixture.view('uint32', new Uint32Array(count));
      const status = fixture.view('uint32', new Uint32Array(count));
      fixture.graph.add(
        new GPUPointCloudNeighborhoods({
          id: `grid-equality-${indexed}`,
          positions: positions.view,
          radius: 1,
          capacity: 4,
          grid: indexed ? {origin: [0, 0, 0], cellSize: 0.5, dimensions: [16, 16, 16]} : undefined,
          indices: indices.view,
          squaredDistances: distances.view,
          counts: counts.view,
          status: status.view
        })
      );
      return {indices, distances, counts, status};
    });
    await fixture.run();
    expect(await readIntegers(results[1].indices.buffer)).toEqual(
      await readIntegers(results[0].indices.buffer)
    );
    expect(await readFloats(results[1].distances.buffer)).toEqual(
      await readFloats(results[0].distances.buffer)
    );
    expect(await readIntegers(results[1].counts.buffer)).toEqual(
      await readIntegers(results[0].counts.buffer)
    );
    expect((await readIntegers(results[1].status.buffer)).at(-1)).toBe(1);
    expect((await readIntegers(results[1].counts.buffer))[0]).toBe(4);
    positions.buffer.write(new Float32Array(count * 3).fill(100));
    await fixture.run();
    expect(await readIntegers(results[1].counts.buffer)).toEqual(Array(count).fill(0));
    expect(await readIntegers(results[1].indices.buffer)).toEqual(
      Array(count * 4).fill(0xffffffff)
    );
    expect(await readIntegers(results[1].status.buffer)).toEqual(Array(count).fill(1));
  } finally {
    fixture.destroy();
  }
});
