// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {addKeplerPointWeights} from '../src/analysis/point-weights';
import {
  buildKeplerStreamedPointWeights,
  importKeplerStreamedPointWeights
} from '../src/analysis/point-weights-streamed';
import {importGraphBuffer} from '../../graph/src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32
} from '../../graph/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

const settings = {
  kind: 'distance-band' as const,
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude' as const,
  metric: 'great-circle' as const,
  units: 'meters' as const,
  radiusMeters: 3000,
  neighborCapacity: 65536
};

test('streamed distance-band chunks list the same neighbors as the whole build plus one self slot per row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 200;
  const longitudes = Float32Array.from(
    {length: rowCount},
    (_, index) => -74 + 0.005 * (index % 20) + 0.0003 * Math.sin(index)
  );
  const latitudes = Float32Array.from(
    {length: rowCount},
    (_, index) => 40.7 + 0.005 * Math.floor(index / 20) + 0.0003 * Math.cos(index)
  );
  const longitudeBuffer = createInputBuffer(device, longitudes);
  const latitudeBuffer = createInputBuffer(device, latitudes);

  // Whole build.
  const graph = new GPUCommandGraph(device);
  const offsets = createOutputBuffer(device, rowCount + 1);
  const neighbors = createOutputBuffer(device, settings.neighborCapacity);
  const weights = createOutputBuffer(device, settings.neighborCapacity);
  const status = [0, 1, 2].map(() => createOutputBuffer(device, 1));
  const pointWeights = addKeplerPointWeights(graph, {
    source: {datasetId: 'points', revision: 1},
    settings,
    longitude: importGraphBuffer(graph, 'longitude', longitudeBuffer, 'float32', rowCount),
    latitude: importGraphBuffer(graph, 'latitude', latitudeBuffer, 'float32', rowCount),
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
  const wholeOffsets = await readUint32(offsets, rowCount + 1);
  const wholeNeighbors = await readUint32(neighbors, settings.neighborCapacity);
  expect((await readUint32(status[0], 1))[0]).toBe(0);
  pointWeights.destroyParameters();

  // Streamed build with chunks forced to a few hundred slots.
  const streamed = await buildKeplerStreamedPointWeights({
    device,
    rowCount,
    firstCoordinate: longitudeBuffer,
    secondCoordinate: latitudeBuffer,
    settings,
    maximumSlotsPerChunk: 600
  });
  expect(streamed.chunks.length).toBeGreaterThan(1);
  expect(streamed.neighborCount).toBe(wholeOffsets[rowCount]);
  for (const chunk of streamed.chunks) {
    const chunkOffsets = await readUint32(chunk.offsets, chunk.rowCount + 1);
    const chunkNeighbors = await readUint32(chunk.neighbors, chunk.slotCount);
    for (let row = 0; row < chunk.rowCount; row++) {
      const globalRow = chunk.rowStart + row;
      const expected = Array.from(
        wholeNeighbors.slice(wholeOffsets[globalRow], wholeOffsets[globalRow + 1])
      );
      const actual = Array.from(chunkNeighbors.slice(chunkOffsets[row], chunkOffsets[row + 1]));
      // The cross join also lists the row itself.
      expect(actual.filter(neighbor => neighbor !== globalRow)).toEqual(expected);
    }
  }
  const consumerGraph = new GPUCommandGraph(device);
  const chunked = importKeplerStreamedPointWeights(consumerGraph, streamed);
  expect(chunked.chunks).toHaveLength(streamed.chunks.length);
  expect(chunked.information?.hasSelfWeights).toBe(true);
  streamed.destroy();
  compiled.destroy();
  for (const buffer of [longitudeBuffer, latitudeBuffer, offsets, neighbors, weights, ...status])
    buffer.destroy();
});
