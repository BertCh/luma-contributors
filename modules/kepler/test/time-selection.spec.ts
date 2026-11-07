// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getInt64TimeWords} from '@luma-contributors/analysis/dataframe-analysis';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {GPUKeplerTimeSelection} from '../src/selection/gpu-time-selection';

it('time selection preserves exact adjacent microseconds beyond Number precision and negative epochs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const epoch = 9_007_199_254_740_993n;
  const timestamps = makeTimeVector(device, [[-2n, -1n, 0n], [], [epoch, epoch + 1n, epoch + 2n]]);
  const validity = makeMaskVector(device, [[1, 0, 1], [], [1, 1, 1]]);
  const readSpies = timestamps.data.map(chunk => vi.spyOn(chunk.buffer, 'readAsync'));
  const selection = new GPUKeplerTimeSelection({
    device,
    timestamps,
    validity,
    unit: 'microsecond',
    window: {start: epoch + 1n, end: epoch + 1n, unit: 'microsecond'}
  });
  const masks = [...selection.mask.data];
  expect(selection.readbackByteLength).toBe(0);
  try {
    submit(device, selection);
    expect(await readMask(selection)).toEqual([0, 0, 0, 0, 1, 0]);
    expect(await selection.readStatus()).toMatchObject({
      revision: 0,
      complete: true,
      selectedCount: 1,
      overflow: 0
    });
    expect(selection.readbackByteLength).toBe(8);
    selection.setWindow({start: epoch, end: epoch + 1n, unit: 'microsecond'});
    expect(await selection.readStatus()).toMatchObject({stale: true, complete: false});
    submit(device, selection);
    expect(await readMask(selection)).toEqual([0, 0, 0, 1, 1, 0]);
    selection.setWindow({start: -2n, end: 0n, unit: 'microsecond'});
    submit(device, selection);
    expect(await readMask(selection)).toEqual([1, 0, 1, 0, 0, 0]);
    selection.clear();
    submit(device, selection);
    expect(await readMask(selection)).toEqual([1, 0, 1, 1, 1, 1]);
    selection.setWindow({start: 1n, end: -1n, unit: 'microsecond'});
    submit(device, selection);
    expect(await readMask(selection)).toEqual([0, 0, 0, 0, 0, 0]);
    expect(selection.mask.data).toEqual(masks);
    expect(await selection.readStatus()).toMatchObject({complete: true, selectedCount: 0});
    for (const spy of readSpies) expect(spy).not.toHaveBeenCalled();
  } finally {
    selection.destroy();
    for (const chunk of timestamps.data) expect(chunk.buffer.destroyed).toBe(false);
    readSpies.forEach(spy => spy.mockRestore());
    timestamps.destroy();
    validity.destroy();
  }
});

it('time selection uses inclusive interval endpoints, both validity masks and lossless host unit conversion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const timestamps = makeTimeVector(device, [[-10_000n, 0n], [], [10_000n, 20_000n]]);
  const endTimestamps = makeTimeVector(device, [[-5_000n, 10_000n], [], [20_000n, 30_000n]]);
  const validity = makeMaskVector(device, [[1, 1], [], [1, 1]]);
  const endValidity = makeMaskVector(device, [[1, 1], [], [1, 1]]);
  const selection = new GPUKeplerTimeSelection({
    device,
    timestamps,
    endTimestamps,
    validity,
    endValidity,
    unit: 'microsecond',
    window: {start: 10, end: 20, unit: 'millisecond'}
  });
  try {
    submit(device, selection);
    expect(await readMask(selection)).toEqual([0, 1, 1, 1]);
    validity.data[2].buffer.write(new Uint32Array([0, 1]));
    endValidity.data[2].buffer.write(new Uint32Array([1, 0]));
    selection.invalidate();
    submit(device, selection);
    expect(await readMask(selection)).toEqual([0, 1, 0, 0]);
    expect(await selection.readStatus()).toMatchObject({
      revision: 1,
      selectedCount: 1,
      complete: true
    });
    selection.setWindow({start: -5, end: -5, unit: 'millisecond'});
    submit(device, selection);
    expect(await readMask(selection)).toEqual([1, 0, 0, 0]);
  } finally {
    selection.destroy();
    for (const vector of [timestamps, endTimestamps, validity, endValidity]) vector.destroy();
  }
});

it('empty time input encodes and publishes an exact zero count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunks of [[], [[]]] as bigint[][][]) {
    const timestamps = makeTimeVector(device, chunks);
    const selection = new GPUKeplerTimeSelection({
      device,
      timestamps,
      unit: 'nanosecond',
      window: {start: -(1n << 63n), end: (1n << 63n) - 1n, unit: 'nanosecond'}
    });
    try {
      submit(device, selection);
      expect(await readMask(selection)).toEqual([]);
      expect(await selection.readStatus()).toMatchObject({
        selectedCount: 0,
        complete: true,
        overflow: 0
      });
    } finally {
      selection.destroy();
      timestamps.destroy();
    }
  }
});

function makeTimeVector(device: Device, chunks: bigint[][]): GPUVector<'uint32x2'> {
  return new GPUVector({
    type: 'data',
    name: 'timestamps',
    format: 'uint32x2',
    data: chunks.map(
      values =>
        new GPUData({
          format: 'uint32x2',
          length: values.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            ...(values.length
              ? {data: getInt64TimeWords(BigInt64Array.from(values))}
              : {byteLength: 8}),
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          })
        })
    )
  });
}

function makeMaskVector(device: Device, chunks: number[][]): GPUVector<'uint32'> {
  return new GPUVector({
    type: 'data',
    name: 'validity',
    format: 'uint32',
    data: chunks.map(
      values =>
        new GPUData({
          format: 'uint32',
          length: values.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            ...(values.length ? {data: Uint32Array.from(values)} : {byteLength: 4}),
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          })
        })
    )
  });
}

function submit(device: Device, selection: GPUKeplerTimeSelection): void {
  const encoder = device.createCommandEncoder();
  selection.encode(encoder);
  device.submit(encoder.finish());
}

async function readMask(selection: GPUKeplerTimeSelection): Promise<number[]> {
  return (
    await Promise.all(
      selection.mask.data.map(async chunk => {
        if (!chunk.length) return [];
        const bytes = await chunk.buffer.readAsync(0, chunk.length * 4);
        return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
      })
    )
  ).flat();
}
