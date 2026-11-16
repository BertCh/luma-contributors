// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNaturalNeighbor} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createPaddedQueryFixture} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const QUERY_POINTS = [
  [0, 0],
  [0.25, 0.25],
  [0.5, 0],
  [0.1, 0.6],
  [1, 0],
  [2, 2]
];
async function runNaturalNeighbor(
  device: Device,
  chunked: boolean,
  indexed: boolean,
  chunkLengths = [1, 3, 2]
) {
  const fixture = createPaddedQueryFixture(device);
  const {graph, input, output} = fixture;
  const positions = input('positions', 'float32x2', new Float32Array([0, 0, 1, 0, 0, 1]));
  const values = input('values', 'float32', new Float32Array([1, 2, 3]));
  const triangles = indexed
    ? input('triangles', 'uint32', new Uint32Array([0, 1, 2, 0xffffffff, 0xffffffff, 0xffffffff]))
    : undefined;
  const total = chunkLengths.reduce((sum, length) => sum + length, 0);
  const lengths = chunked ? chunkLengths : [total];
  const queries: GraphDataView<'float32x2'>[] = [];
  const seeds: GraphDataView<'uint32'>[] = [];
  const predictions: GraphDataView<'float32x4'>[] = [];
  const statuses: GraphDataView<'uint32'>[] = [];
  const weights: GraphDataView<'float32' | 'float32x2'>[] = [];
  let rowBase = 0;
  for (const [chunkIndex, length] of lengths.entries()) {
    queries.push(
      input(
        `queries-${chunkIndex}`,
        'float32x2',
        new Float32Array(Array.from({length}, (_, row) => QUERY_POINTS[(rowBase + row) % 6]).flat())
      )
    );
    if (indexed)
      seeds.push(
        input(
          `seeds-${chunkIndex}`,
          'uint32',
          new Uint32Array(
            Array.from({length}, (_, row) => ((rowBase + row) % 6 === 5 ? 0xffffffff : 0))
          )
        )
      );
    predictions.push(output(`predictions-${chunkIndex}`, 'float32x4', length));
    statuses.push(output(`statuses-${chunkIndex}`, 'uint32', length));
    weights.push(output(`weights-${chunkIndex}`, indexed ? 'float32x2' : 'float32', length * 3));
    rowBase += length;
  }
  graph.add(
    new GPUNaturalNeighbor({
      positions,
      values,
      triangles,
      maximumNeighborCount: 3,
      queries: chunked ? createVectorView('queries', 'float32x2', queries) : queries[0],
      queryTriangles: indexed
        ? chunked
          ? createVectorView('seeds', 'uint32', seeds)
          : seeds[0]
        : undefined,
      output: {
        values: chunked
          ? createVectorView('predictions', 'float32x4', predictions)
          : predictions[0],
        status: chunked ? createVectorView('statuses', 'uint32', statuses) : statuses[0],
        weights: chunked
          ? createVectorView('weights', indexed ? 'float32x2' : 'float32', weights)
          : weights[0]
      }
    })
  );
  const compiled = graph.compile();
  try {
    return await fixture.submitTwice(compiled, async () => ({
      values: await fixture.readRows(predictions),
      statuses: await fixture.readRows(statuses),
      weights: await fixture.readRows(weights)
    }));
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
}

for (const indexed of [false, true])
  it(`matches packed values/status/weights and triangle barycentric coordinates (indexed=${indexed})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const packed = await runNaturalNeighbor(device, false, indexed);
    const chunked = await runNaturalNeighbor(device, true, indexed);
    expect(chunked).toEqual(packed);
    expect(chunked.statuses).toEqual([0, 0, 0, 0, 0, 1]);
    for (const [queryIndex, [x, y]] of QUERY_POINTS.slice(0, 5).entries()) {
      expect(chunked.values[queryIndex * 4]).toBeCloseTo(1 + x + 2 * y, 4);
      const actualWeights = [0, 0, 0];
      for (let slot = 0; slot < 3; slot++) {
        if (indexed) {
          const sampleIndex = chunked.weights[(queryIndex * 3 + slot) * 2];
          if (sampleIndex >= 0)
            actualWeights[sampleIndex] = chunked.weights[(queryIndex * 3 + slot) * 2 + 1];
        } else actualWeights[slot] = chunked.weights[queryIndex * 3 + slot];
      }
      for (const [sampleIndex, expected] of [1 - x - y, x, y].entries())
        expect(actualWeights[sampleIndex]).toBeCloseTo(expected, 4);
    }
  });

it('writes 600000 indexed queries beyond the old packed 32-neighbor weight binding limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    data: Float32Array | Uint32Array,
    length: number
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const rowCount = 600_000;
  const chunkLength = 200_000;
  const neighborCount = 32;
  const queries = input('queries', 'float32x2', new Float32Array(rowCount * 2), rowCount);
  const seeds = input('seeds', 'uint32', new Uint32Array(rowCount), rowCount);
  const predictionBuffer = createOutputBuffer(device, rowCount * 4);
  const statusBuffer = createOutputBuffer(device, rowCount);
  const weightBuffer = createOutputBuffer(device, rowCount * neighborCount * 2);
  buffers.push(predictionBuffer, statusBuffer, weightBuffer);
  const predictions = importGraphBuffer(
    graph,
    'predictions',
    predictionBuffer,
    'float32x4',
    rowCount
  );
  const statuses = importGraphBuffer(graph, 'statuses', statusBuffer, 'uint32', rowCount);
  const weights = importGraphBuffer(
    graph,
    'weights',
    weightBuffer,
    'float32x2',
    rowCount * neighborCount
  );
  const split = <Format extends 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    view: GraphDataView<Format>,
    width = 1
  ) =>
    createVectorView(
      name,
      view.format,
      [0, 1, 2].map(index =>
        graph.createDataView(view.buffer, {
          format: view.format,
          length: chunkLength * width,
          byteOffset: index * chunkLength * width * view.rowByteLength
        })
      )
    );
  graph.add(
    new GPUNaturalNeighbor({
      positions: input('positions', 'float32x2', new Float32Array([0, 0, 1, 0, 0, 1]), 3),
      values: input('values', 'float32', new Float32Array([1, 2, 3]), 3),
      triangles: input(
        'triangles',
        'uint32',
        new Uint32Array([0, 1, 2, 0xffffffff, 0xffffffff, 0xffffffff]),
        6
      ),
      queries: split('queries', queries),
      queryTriangles: split('seeds', seeds),
      maximumNeighborCount: neighborCount,
      output: {
        values: split('predictions', predictions),
        status: split('statuses', statuses),
        weights: split('weights', weights, neighborCount)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const row of [0, chunkLength - 1, chunkLength, rowCount - 1]) {
      const predictionBytes = await predictionBuffer.readAsync(row * 16, 16);
      expect(
        Array.from(new Float32Array(predictionBytes.buffer, predictionBytes.byteOffset, 4))
      ).toEqual([1, 0, 1, 1]);
      const weightBytes = await weightBuffer.readAsync(row * neighborCount * 8, 16);
      expect(Array.from(new Float32Array(weightBytes.buffer, weightBytes.byteOffset, 4))).toEqual([
        0, 1, -1, 0
      ]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

for (const indexed of [false, true])
  it(`preserves offsets and padding across 1/2/3-row dispatches (indexed=${indexed})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const lengths = [1, 2, 3];
    const packed = await runNaturalNeighbor(device, false, indexed, lengths);
    const chunked = await runNaturalNeighbor(device, true, indexed, lengths);
    expect(chunked).toEqual(packed);
    expect(chunked.statuses).toEqual([0, 0, 0, 0, 0, 1]);
    for (let row = 0; row < 5; row++) {
      const [x, y] = QUERY_POINTS[row];
      expect(chunked.values[row * 4]).toBeCloseTo(1 + x + 2 * y, 4);
    }
  });
