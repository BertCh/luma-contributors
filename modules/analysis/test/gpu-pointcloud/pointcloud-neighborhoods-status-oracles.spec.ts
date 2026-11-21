// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudNeighborhoods,
  getGPUPointCloudNeighborhoodsParameterValues
} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';

for (const strategy of ['serial', 'cooperative'] as const) {
  it(`neighborhood status writes complete words and clears all three bits on replay: ${strategy}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      // Source x coordinates are [-1, 2, 0, 1]. At radius 1 the two interior
      // queries have two neighbors, exceeding capacity 1 (bit 2). Endpoints
      // have one neighbor but their spheres exceed [-1,2] (bit 4).
      // The out-of-range row has no position and must have only bit 1.
      const positions = fixture.view(
        'float32x3',
        new Float32Array([-1, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0])
      );
      const queryRows = fixture.view('uint32', new Uint32Array([2, 3, 0, 1, 99]));
      const indices = fixture.view('uint32', new Uint32Array(5));
      const squaredDistances = fixture.view('float32', new Float32Array(5));
      const counts = fixture.view('uint32', new Uint32Array(5).fill(0xffffffff));
      const status = fixture.view('uint32', new Uint32Array(5).fill(0xffffffff));
      fixture.graph.add(
        new GPUPointCloudNeighborhoods({
          positions: positions.view,
          queryRows: queryRows.view,
          radius: 1,
          capacity: 1,
          strategy,
          supportBounds: {minimum: [-1, -2, -2], maximum: [2, 2, 2]},
          output: {
            indices: indices.view,
            squaredDistances: squaredDistances.view,
            counts: counts.view,
            status: status.view
          }
        })
      );
      await fixture.run();
      expect(await readIntegers(counts.buffer)).toEqual([2, 2, 1, 1, 0]);
      expect(await readIntegers(status.buffer)).toEqual([2, 2, 4, 4, 1]);

      // Four tetrahedron vertices at pairwise distance sqrt(2) have no
      // other point within radius 1. Each radius sphere is inside the
      // certified bounds. Every earlier flag must clear, including bit 1.
      positions.buffer.write(new Float32Array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]));
      queryRows.buffer.write(new Uint32Array([0, 1, 2, 3, 0]));
      await fixture.run();
      expect(await readIntegers(counts.buffer)).toEqual([0, 0, 0, 0, 0]);
      expect(await readIntegers(status.buffer)).toEqual([0, 0, 0, 0, 0]);
    } finally {
      fixture.destroy();
    }
  });
}

for (const strategy of ['serial', 'cooperative'] as const) {
  for (const runtimeBounds of [false, true]) {
    it(`review 2 R1 outside-grid point retains support bit, missing row does not: ${strategy}, runtime=${runtimeBounds}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        const coordinates = new Float32Array(257 * 3);
        coordinates.set([50, 0, 0], 256 * 3);
        const positions = fixture.view('float32x3', coordinates);
        const queryRows = fixture.view('uint32', new Uint32Array([256, 257]));
        const indices = fixture.view('uint32', new Uint32Array(2));
        const squaredDistances = fixture.view('float32', new Float32Array(2));
        const counts = fixture.view('uint32', new Uint32Array([99, 99]));
        const status = fixture.view('uint32', new Uint32Array([99, 99]));
        const settings = {
          radius: 2,
          supportBounds: {minimum: [-5, -5, -5] as const, maximum: [5, 5, 5] as const}
        };
        fixture.graph.add(
          new GPUPointCloudNeighborhoods({
            ...settings,
            strategy,
            capacity: 1,
            positions: positions.view,
            queryRows: queryRows.view,
            grid: {origin: [-5, -5, -5], dimensions: [10, 10, 10], cellSize: 1},
            parameters: runtimeBounds
              ? fixture.view('float32', getGPUPointCloudNeighborhoodsParameterValues(settings)).view
              : undefined,
            output: {
              indices: indices.view,
              squaredDistances: squaredDistances.view,
              counts: counts.view,
              status: status.view
            }
          })
        );
        await fixture.run();
        // Row 256 exists but is outside both grid and certified sphere bounds.
        // Row 257 has no sphere. Neither query searches any indexed candidate.
        expect(await readIntegers(counts.buffer)).toEqual([0, 0]);
        expect(await readIntegers(status.buffer)).toEqual([5, 1]);
      } finally {
        fixture.destroy();
      }
    });
    it(`R2 nonexistent query status is independent of source ordering: ${strategy}, runtime=${runtimeBounds}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        const positions = fixture.view('float32x3', new Float32Array(12));
        const queryRows = fixture.view('uint32', new Uint32Array([99]));
        const indices = fixture.view('uint32', new Uint32Array(1));
        const squaredDistances = fixture.view('float32', new Float32Array(1));
        const counts = fixture.view('uint32', new Uint32Array([99]));
        const status = fixture.view('uint32', new Uint32Array([99]));
        const settings = {
          radius: 1,
          supportBounds: {minimum: [-1, -2, -2] as const, maximum: [2, 2, 2] as const}
        };
        const parameters = runtimeBounds
          ? fixture.view('float32', getGPUPointCloudNeighborhoodsParameterValues(settings)).view
          : undefined;
        fixture.graph.add(
          new GPUPointCloudNeighborhoods({
            ...settings,
            positions: positions.view,
            queryRows: queryRows.view,
            capacity: 1,
            strategy,
            parameters,
            output: {
              indices: indices.view,
              squaredDistances: squaredDistances.view,
              counts: counts.view,
              status: status.view
            }
          })
        );
        for (const coordinates of [
          [-1, 2, 0, 1],
          [-1, 1, 0, 2]
        ]) {
          positions.buffer.write(
            Float32Array.from(coordinates.flatMap(horizontal => [horizontal, 0, 0]))
          );
          // Row 99 is not one of four rows; it has no neighbors and no
          // position whose support sphere could add bit 4. Whole code 1.
          await fixture.run();
          expect(await readIntegers(counts.buffer)).toEqual([0]);
          expect(await readIntegers(status.buffer)).toEqual([1]);
        }
      } finally {
        fixture.destroy();
      }
    });
  }
}
