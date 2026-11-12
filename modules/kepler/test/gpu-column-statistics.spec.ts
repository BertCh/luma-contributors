// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {readGPUColumnStatistics} from '../src/exploration/gpu-column-statistics';

test('column statistics over chunked values, validity and selection match a CPU reduction', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for column statistics verification');
  const chunks = [[4, -2, 7.5], [], [1, 3, 9, -6, 0.5]];
  const validityRows = [[1, 0, 1], [], [1, 1, 0, 1, 1]];
  const selectionRows = [[1, 1, 1], [], [0, 1, 1, 1, 1]];
  const values = vector(
    device,
    'float32',
    chunks.map(chunk => new Float32Array(chunk))
  );
  const validity = vector(
    device,
    'uint32',
    validityRows.map(chunk => new Uint32Array(chunk))
  );
  const selection = vector(
    device,
    'uint32',
    selectionRows.map(chunk => new Uint32Array(chunk))
  );
  try {
    const flatValues = chunks.flat();
    const flatValidity = validityRows.flat();
    const flatSelection = selectionRows.flat();
    const populationRows = flatSelection.filter(Boolean).length;
    const counted = flatValues.filter((_, row) => flatValidity[row] && flatSelection[row]);
    const sum = counted.reduce((total, value) => total + value, 0);
    const mean = sum / counted.length;
    const squares = counted.reduce((total, value) => total + (value - mean) ** 2, 0);
    const statistics = await readGPUColumnStatistics({
      device,
      values: values as GPUVector<'float32'>,
      validity: validity as GPUVector<'uint32'>,
      selection: selection as GPUVector<'uint32'>,
      populationRows,
      histogramBinCount: 4,
      isCurrent: () => true
    });
    expect(statistics).not.toBeNull();
    expect(statistics!.populationRows).toBe(populationRows);
    expect(statistics!.count).toBe(counted.length);
    expect(statistics!.nullCount).toBe(populationRows - counted.length);
    expect(statistics!.minimum).toBe(Math.min(...counted));
    expect(statistics!.maximum).toBe(Math.max(...counted));
    expect(statistics!.sum).toBeCloseTo(sum, 5);
    expect(statistics!.mean).toBeCloseTo(mean, 5);
    expect(statistics!.sampleVariance).toBeCloseTo(squares / (counted.length - 1), 4);
    expect(statistics!.histogram).toHaveLength(4);
    expect(statistics!.histogram.reduce((total, count) => total + count, 0)).toBe(counted.length);

    const stale = await readGPUColumnStatistics({
      device,
      values: values as GPUVector<'float32'>,
      populationRows: flatValues.length,
      isCurrent: () => false
    });
    expect(stale).toBeNull();
  } finally {
    values.destroy();
    validity.destroy();
    selection.destroy();
  }
});

function vector(
  device: Device,
  format: 'float32' | 'uint32',
  chunks: (Float32Array | Uint32Array)[]
): GPUVector {
  return new GPUVector({
    name: format,
    type: 'data',
    format,
    ownsData: true,
    data: chunks.map(
      chunk =>
        new GPUData({
          buffer: device.createBuffer({
            data: chunk.length ? chunk : undefined,
            byteLength: Math.max(4, chunk.byteLength),
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          }),
          format,
          length: chunk.length,
          ownsBuffer: true
        })
    )
  });
}
