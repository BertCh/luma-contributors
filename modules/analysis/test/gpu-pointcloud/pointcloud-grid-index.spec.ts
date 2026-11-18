// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createPointCloudGridIndex} from '../../src/pointcloud-analysis/operations/pointcloud-grid-index';
import {getPointCloudKernelNodes} from '../../src/pointcloud-analysis/operations/pointcloud-kernel-nodes';
import {makeSpatialFixture, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

for (const [pointCount, cellCount] of [
  [0, 1],
  [1, 8],
  [255, 8],
  [256, 8],
  [257, 8],
  [2051, 1],
  [2051, 15],
  [2051, 16],
  [2051, 255],
  [2051, 256],
  [2051, 257]
]) {
  it(`grid index preserves stable IDs and offsets for ${pointCount} points in ${cellCount} cells`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    const positions = new Float32Array(pointCount * 3);
    const writePositions = (mode: number) => {
      for (let index = 0; index < pointCount; index++) {
        let coordinate = ((index * 17 + mode) % cellCount) + 0.25;
        // Invalid rows deliberately precede the highest valid cell, exposing sentinel collisions.
        if (mode === 2 || index % 11 === 0) coordinate = cellCount;
        else if (index % 11 === 1) coordinate = cellCount - 0.25;
        else if (index % 11 === 2) coordinate = Number.NaN;
        else if (index % 11 === 3) coordinate = Number.POSITIVE_INFINITY;
        else if (index % 11 === 4) coordinate = -0.25;
        positions.set([coordinate, 0.25, 0.25], index * 3);
      }
    };
    writePositions(0);
    const input = fixture.view('float32x3', positions);
    const indices = fixture.view('uint32', new Uint32Array(pointCount));
    const offsets = fixture.view('uint32', new Uint32Array(cellCount + 1));
    const built = createPointCloudGridIndex(fixture.graph, 'index', input.view, {
      origin: [0, 0, 0],
      dimensions: [cellCount, 1, 1],
      cellSize: 1
    });
    for (const node of built.nodes) fixture.graph.add(node);
    for (const node of getPointCloudKernelNodes(fixture.graph, {
      id: 'capture-indices',
      length: pointCount,
      inputs: {sortedIndices: built.sortedIndices},
      outputs: {indices: indices.view},
      body: 'writeIndices(index,readSortedIndices(index));'
    }))
      fixture.graph.add(node);
    for (const node of getPointCloudKernelNodes(fixture.graph, {
      id: 'capture-offsets',
      length: cellCount + 1,
      inputs: {gridOffsets: built.offsets},
      outputs: {offsets: offsets.view},
      body: 'writeOffsets(index,readGridOffsets(index));'
    }))
      fixture.graph.add(node);
    try {
      for (const mode of [0, 1, 2, 0]) {
        writePositions(mode);
        if (pointCount) input.buffer.write(positions);
        await fixture.run();
        const expected = Array.from({length: pointCount}, (_, index) => {
          const coordinate = positions[index * 3];
          const key =
            Number.isFinite(coordinate) && coordinate >= 0 && coordinate < cellCount
              ? Math.floor(coordinate)
              : 0xffffffff;
          return {key, index};
        }).sort((first, second) => first.key - second.key || first.index - second.index);
        expect((await readIntegers(indices.buffer)).slice(0, pointCount)).toEqual(
          expected.map(item => item.index)
        );
        expect(await readIntegers(offsets.buffer)).toEqual(
          Array.from(
            {length: cellCount + 1},
            (_, cell) => expected.filter(item => item.key < cell).length
          )
        );
      }
    } finally {
      fixture.destroy();
    }
  });
}
