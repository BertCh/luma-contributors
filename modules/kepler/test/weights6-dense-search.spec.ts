// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {addKeplerPointWeights} from '../src/analysis/point-weights';
import {importGraphBuffer} from '../../graph/src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32
} from '../../graph/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

const rowCount = 300;
const settings = {
  kind: 'distance-band' as const,
  xColumn: 'x',
  yColumn: 'y',
  coordinateSpace: 'planar' as const,
  metric: 'native' as const,
  units: 'meters' as const,
  crs: 'EPSG:3857',
  bounds: [0, 0, 400000, 100000] as const,
  radiusMeters: 500,
  neighborCapacity: 65536
};

/** Strongly clustered projected points: three tight clusters in a wide extent. */
function createClusteredCoordinates() {
  const x = new Float32Array(rowCount);
  const y = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    const cluster = row % 3;
    x[row] = 50000 + cluster * 150000 + (row % 17) * 40 + Math.sin(row) * 20;
    y[row] = 30000 + cluster * 20000 + (row % 13) * 40 + Math.cos(row) * 20;
  }
  return {x, y};
}

async function buildOffsetsAndNeighbors(searchStructure: 'grid' | 'bvh') {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {x, y} = createClusteredCoordinates();
  const graph = new GPUCommandGraph(device);
  const offsets = createOutputBuffer(device, rowCount + 1);
  const neighbors = createOutputBuffer(device, settings.neighborCapacity);
  const weights = createOutputBuffer(device, settings.neighborCapacity);
  const status = [0, 1, 2].map(() => createOutputBuffer(device, 1));
  const pointWeights = addKeplerPointWeights(graph, {
    source: {datasetId: 'clustered', revision: 1},
    settings,
    searchStructure,
    x: importGraphBuffer(graph, 'x', createInputBuffer(device, x), 'float32', rowCount),
    y: importGraphBuffer(graph, 'y', createInputBuffer(device, y), 'float32', rowCount),
    output: {
      offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', rowCount + 1),
      neighbors: importGraphBuffer(
        graph,
        'neighbors',
        neighbors,
        'uint32',
        settings.neighborCapacity
      ),
      weights: importGraphBuffer(graph, 'weights', weights, 'float32', settings.neighborCapacity),
      status: {
        overflow: importGraphBuffer(graph, 'overflow', status[0], 'uint32', 1),
        requiredCount: importGraphBuffer(graph, 'required', status[1], 'uint32', 1),
        invalidCount: importGraphBuffer(graph, 'invalid', status[2], 'uint32', 1)
      }
    }
  });
  const compiled = graph.compile();
  const encoder = device.createCommandEncoder();
  compiled.encode(encoder, {parameters: undefined} as never);
  device.submit(encoder.finish());
  const resultOffsets = await readUint32(offsets, rowCount + 1);
  const resultNeighbors = await readUint32(neighbors, settings.neighborCapacity);
  expect((await readUint32(status[0], 1))[0]).toBe(0);
  pointWeights.destroyParameters();
  compiled.destroy();
  return {offsets: Array.from(resultOffsets), neighbors: Array.from(resultNeighbors)};
}

test('BVH and grid searches list identical exact neighbors on clustered projected points', async () => {
  const grid = await buildOffsetsAndNeighbors('grid');
  const bvh = await buildOffsetsAndNeighbors('bvh');
  expect(grid.offsets[rowCount]).toBeGreaterThan(rowCount);
  expect(bvh.offsets).toEqual(grid.offsets);
  expect(bvh.neighbors.slice(0, grid.offsets[rowCount])).toEqual(
    grid.neighbors.slice(0, grid.offsets[rowCount])
  );
});
