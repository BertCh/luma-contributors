// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULineLocate} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-line-locate';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createPaddedQueryFixture} from './gpu-linear-referencing-chunks.test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function runLocate(device: Device, chunked: boolean, chunkLengths = [1, 3, 2]) {
  const fixture = createPaddedQueryFixture(device);
  const {graph, input, output} = fixture;
  const positions = input('positions', 'float32x2', new Float32Array([0, 0, 10, 0]));
  const lineOffsets = input('line-offsets', 'uint32', new Uint32Array([0, 2]));
  const parameters = input('parameters', 'float32', new Float32Array([1, 0, 0, 0]));
  const measures = [-1, 0, 5, 10, 15, Number.NaN];
  const offsets = [0, 1, -1, 2, 0, 0];
  const total = chunkLengths.reduce((sum, length) => sum + length, 0);
  const lengths = chunked ? chunkLengths : [total];
  const eventChunks: GraphDataView<'uint32'>[] = [];
  const measureChunks: GraphDataView<'float32'>[] = [];
  const offsetChunks: GraphDataView<'float32'>[] = [];
  const positionChunks: GraphDataView<'float32x2'>[] = [];
  const statusChunks: GraphDataView<'uint32'>[] = [];
  let rowBase = 0;
  for (const [chunkIndex, length] of lengths.entries()) {
    eventChunks.push(input(`events-${chunkIndex}`, 'uint32', new Uint32Array(length)));
    measureChunks.push(
      input(
        `measures-${chunkIndex}`,
        'float32',
        new Float32Array(Array.from({length}, (_, row) => measures[(rowBase + row) % 6]))
      )
    );
    offsetChunks.push(
      input(
        `offsets-${chunkIndex}`,
        'float32',
        new Float32Array(Array.from({length}, (_, row) => offsets[(rowBase + row) % 6]))
      )
    );
    positionChunks.push(output(`positions-${chunkIndex}`, 'float32x2', length));
    statusChunks.push(output(`statuses-${chunkIndex}`, 'uint32', length));
    rowBase += length;
  }
  graph.add(
    new GPULineLocate({
      positions,
      lineOffsets,
      parameters,
      eventLineIndices: chunked
        ? createVectorView('events', 'uint32', eventChunks)
        : eventChunks[0],
      eventMeasures: chunked
        ? createVectorView('measures', 'float32', measureChunks)
        : measureChunks[0],
      eventOffsets: chunked
        ? createVectorView('offsets', 'float32', offsetChunks)
        : offsetChunks[0],
      output: {
        positions: chunked
          ? createVectorView('output', 'float32x2', positionChunks)
          : positionChunks[0],
        statuses: chunked ? createVectorView('status', 'uint32', statusChunks) : statusChunks[0]
      }
    })
  );
  const compiled = graph.compile();
  try {
    return await fixture.submitTwice(compiled, async () => ({
      positions: await fixture.readRows(positionChunks),
      statuses: await fixture.readRows(statusChunks)
    }));
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
}

it('matches packed events exactly for three uneven chunks and the horizontal-line oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const packed = await runLocate(device, false);
  const chunked = await runLocate(device, true);
  expect(chunked).toEqual(packed);
  expect(chunked.positions).toEqual([0, 0, 0, 1, 5, -1, 10, 2, 10, 0, Number.NaN, Number.NaN]);
  expect(chunked.statuses).toEqual([1, 0, 0, 0, 1, 2]);
});

it('locates six million events beyond the old packed scratch limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array,
    length: number
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const rowCount = 6_000_000;
  const chunkLength = 2_000_000;
  const events = input('events', 'uint32', new Uint32Array(rowCount), rowCount);
  const measures = input('measures', 'float32', new Float32Array(rowCount).fill(5), rowCount);
  const outputBuffer = createOutputBuffer(device, rowCount * 2);
  buffers.push(outputBuffer);
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'float32x2', rowCount);
  const split = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    view: GraphDataView<Format>
  ) =>
    createVectorView(
      name,
      view.format,
      [0, 1, 2].map(index =>
        graph.createDataView(view.buffer, {
          format: view.format,
          length: chunkLength,
          byteOffset: index * chunkLength * view.rowByteLength
        })
      )
    );
  graph.add(
    new GPULineLocate({
      positions: input('positions', 'float32x2', new Float32Array([0, 0, 10, 0]), 2),
      lineOffsets: input('line-offsets', 'uint32', new Uint32Array([0, 2]), 2),
      eventLineIndices: split('events', events),
      eventMeasures: split('measures', measures),
      output: {positions: split('output', output)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const row of [0, chunkLength - 1, chunkLength, 2 * chunkLength, rowCount - 1]) {
      const bytes = await outputBuffer.readAsync(row * 8, 8);
      expect(Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, 2))).toEqual([5, 0]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('preserves offsets and padding across 255/256/257-row event dispatches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lengths = [255, 256, 257];
  const packed = await runLocate(device, false, lengths);
  const chunked = await runLocate(device, true, lengths);
  expect(chunked).toEqual(packed);
  const expected = [
    [0, 0],
    [0, 1],
    [5, -1],
    [10, 2],
    [10, 0],
    [Number.NaN, Number.NaN]
  ];
  for (let row = 0; row < 768; row++) {
    expect(chunked.positions.slice(row * 2, row * 2 + 2)).toEqual(expected[row % 6]);
    expect(chunked.statuses[row]).toBe([1, 0, 0, 0, 1, 2][row % 6]);
  }
});
