// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPURasterRegistration} from '../../../src/raster-analysis/remote-sensing/gpu-raster-registration';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

// Independent two-pass, float64 Pearson correlation of explicitly paired cells. No Welford or
// reduction-tree code is shared with the contributor. Only integer coarse shifts are needed.
function computeCorrelation(
  reference: Float32Array,
  moving: Float32Array,
  referenceMask: Uint32Array,
  movingMask: Uint32Array,
  width: number,
  height: number,
  columnShift: number,
  rowShift: number
) {
  const pairs: [number, number][] = [];
  for (let row = 0; row < height; row++)
    for (let column = 0; column < width; column++) {
      const cell = row * width + column;
      const movingColumn = column - columnShift;
      const movingRow = row - rowShift;
      if (movingColumn < 0 || movingColumn >= width || movingRow < 0 || movingRow >= height)
        continue;
      const source = movingRow * width + movingColumn;
      if (
        referenceMask[cell] &&
        movingMask[source] &&
        Number.isFinite(reference[cell]) &&
        Number.isFinite(moving[source])
      )
        pairs.push([reference[cell], moving[source]]);
    }
  const referenceMean = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
  const movingMean = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;
  let covariance = 0;
  let referenceVariance = 0;
  let movingVariance = 0;
  for (const [first, second] of pairs) {
    covariance += (first - referenceMean) * (second - movingMean);
    referenceVariance += (first - referenceMean) ** 2;
    movingVariance += (second - movingMean) ** 2;
  }
  return {count: pairs.length, score: covariance / Math.sqrt(referenceVariance * movingVariance)};
}

it('GPU-01 reduces masked means and refinement across 65535 workgroups with exact support and packed output sentinels', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const width = 129;
  const height = 129;
  const count = width * height;
  const reference = new Float32Array(count);
  let randomState = 12345;
  for (let cell = 0; cell < count; cell++) {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    reference[cell] = 100 + (randomState % 8192) / 64;
  }
  const moving = Float32Array.from(reference, (_, cell) => {
    const column = (cell % width) + 2;
    const row = Math.floor(cell / width) - 1;
    return column < width && row >= 0 ? 3 * reference[row * width + column] + 11 : NaN;
  });
  const referenceMask = Uint32Array.from(reference, (_, cell) => (cell % 19 !== 0 ? 1 : 0));
  const movingMask = Uint32Array.from(reference, (_, cell) => (cell % 23 !== 0 ? 1 : 0));
  for (let cell = 71; cell < count; cell += 257) reference[cell] = NaN;
  for (let cell = 73; cell < count; cell += 263) moving[cell] = NaN;
  const expected = computeCorrelation(
    reference,
    moving,
    referenceMask,
    movingMask,
    width,
    height,
    2,
    -1
  );
  expect(expected.count).toBeGreaterThan(14000);
  expect(expected.score).toBeCloseTo(1, 12);
  let secondary = 0;
  for (let row = -2; row <= 2; row++)
    for (let column = -3; column <= 3; column++) {
      if (Math.abs(column - 2) <= 1 && Math.abs(row + 1) <= 1) continue;
      secondary = Math.max(
        secondary,
        computeCorrelation(reference, moving, referenceMask, movingMask, width, height, column, row)
          .score
      );
    }
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    values: Float32Array | Uint32Array,
    format: Format,
    prefix: number
  ) => {
    const padded =
      format === 'float32'
        ? new Float32Array(prefix + count + 3).fill(-12345)
        : new Uint32Array(prefix + count + 3).fill(0xa5a5a5a5);
    padded.set(values, prefix);
    const buffer = createInputBuffer(device, padded);
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id: `reduction-input-${serial++}`, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, byteOffset: prefix * 4, length: count});
  };
  const outputBuffer = createOutputBuffer(device, 320);
  buffers.push(outputBuffer);
  outputBuffer.write(new Uint32Array(320).fill(0xa5a5a5a5));
  const handle = graph.importBuffer(
    {id: 'reduction-packed-output', byteLength: outputBuffer.byteLength, usage: outputBuffer.usage},
    outputBuffer
  );
  const output = {
    shift: graph.createDataView(handle, {format: 'float32x2', byteOffset: 4, length: 1}),
    peak: graph.createDataView(handle, {format: 'float32', byteOffset: 264, length: 1}),
    status: graph.createDataView(handle, {format: 'uint32', byteOffset: 524, length: 1}),
    overlap: graph.createDataView(handle, {format: 'uint32', byteOffset: 784, length: 1}),
    // Ratio and peak use separate dispatches and may share an aligned binding window.
    secondPeakRatio: graph.createDataView(handle, {format: 'float32', byteOffset: 268, length: 1})
  };
  try {
    graph.add(
      new GPURasterRegistration({
        mode: 'masked-linear',
        grid: makeGPURasterMetadata({width, height}),
        reference: input(reference, 'float32', 5),
        moving: input(moving, 'float32', 9),
        referenceMask: input(referenceMask, 'uint32', 13),
        movingMask: input(movingMask, 'uint32', 17),
        maximumShift: [3, 2],
        upsampleFactor: 32,
        output
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const words = Uint32Array.from(await readUint32(outputBuffer, 320));
      const floats = new Float32Array(words.buffer);
      expect(floats[1]).toBeCloseTo(2, 5);
      expect(floats[2]).toBeCloseTo(-1, 5);
      expect(floats[66]).toBeCloseTo(expected.score, 5);
      expect(floats[67]).toBeCloseTo(secondary / expected.score, 4);
      expect(words[131]).toBe(0);
      expect(words[196]).toBe(expected.count);
      const written = new Set([1, 2, 66, 67, 131, 196]);
      for (let word = 0; word < words.length; word++)
        if (!written.has(word)) expect(words[word], `untouched word ${word}`).toBe(0xa5a5a5a5);
    } finally {
      compiled.destroy();
    }
  } finally {
    buffers.forEach(buffer => buffer.destroy());
  }
}, 60000);
