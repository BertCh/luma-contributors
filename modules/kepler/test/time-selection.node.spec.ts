// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {
  convertKeplerTimeInstant,
  GPUKeplerTimeSelection
} from '../src/selection/gpu-time-selection';

describe('Kepler exact time units', () => {
  it('preserves integers beyond Number precision and converts units without rounding', () => {
    const instant = 9_007_199_254_740_993n;
    expect(convertKeplerTimeInstant(instant, 'microsecond', 'microsecond')).toBe(instant);
    expect(convertKeplerTimeInstant(-1, 'millisecond', 'microsecond')).toBe(-1000n);
    expect(convertKeplerTimeInstant(-1000n, 'microsecond', 'millisecond')).toBe(-1n);
    expect(convertKeplerTimeInstant(1, 'second', 'nanosecond')).toBe(1_000_000_000n);
    expect(convertKeplerTimeInstant(-(1n << 63n), 'nanosecond', 'nanosecond')).toBe(-(1n << 63n));
  });

  it('rejects unsafe numbers, fractions, lossy conversions and Int64 overflow', () => {
    for (const value of [Number.MAX_SAFE_INTEGER + 1, Infinity, NaN, 0.5, -0.5]) {
      expect(() => convertKeplerTimeInstant(value, 'microsecond', 'microsecond')).toThrow(
        /safe integer/
      );
    }
    for (const value of [1001n, -1001n]) {
      expect(() => convertKeplerTimeInstant(value, 'microsecond', 'millisecond')).toThrow(
        /lose precision/
      );
    }
    expect(() => convertKeplerTimeInstant(1n << 63n, 'microsecond', 'microsecond')).toThrow(
      /Int64/
    );
    expect(() => convertKeplerTimeInstant(9_223_372_037n, 'second', 'nanosecond')).toThrow(/Int64/);
  });
});

describe('GPUKeplerTimeSelection lifecycle', () => {
  it('preserves empty source chunks, borrows sources and owns renderable masks', async () => {
    const device = createNullWebGPUDevice();
    const timestamps = makeVector(device, 'uint32x2', [2, 0, 3]);
    const validity = makeVector(device, 'uint32', [2, 0, 3]);
    const selection = new GPUKeplerTimeSelection({
      device,
      timestamps,
      validity,
      unit: 'microsecond',
      window: {start: 0n, end: 1000n, unit: 'microsecond'}
    });
    const masks = selection.mask.data.map(chunk => chunk.buffer);
    expect(selection.mask.data.map(chunk => chunk.length)).toEqual([2, 0, 3]);
    expect(
      masks.every(
        buffer =>
          (buffer.usage & (Buffer.VERTEX | Buffer.STORAGE)) === (Buffer.VERTEX | Buffer.STORAGE)
      )
    ).toBe(true);
    expect(selection.setWindow({start: -1, end: 1, unit: 'millisecond'})).toBe(1);
    expect(() => selection.setWindow({start: 1n, end: 1001n, unit: 'nanosecond'})).toThrow(
      /precision/
    );
    expect(selection.clear()).toBe(2);
    await expect(selection.readStatus()).rejects.toThrow(/Encode/);
    timestamps.data.push(timestamps.data[0]);
    expect(() => selection.encode({device} as CommandEncoder)).toThrow(/topology changed/);
    timestamps.data.pop();
    selection.destroy();
    selection.destroy();
    expect(masks.every(buffer => buffer.destroyed)).toBe(true);
    expect(selection.accountedResidentByteLength).toBe(0);
    expect(timestamps.data.every(chunk => !chunk.buffer.destroyed)).toBe(true);
    expect(validity.data.every(chunk => !chunk.buffer.destroyed)).toBe(true);
    expect(() => selection.clear()).toThrow(/destroyed/);
    timestamps.destroy();
    validity.destroy();
    device.destroy();
  });

  it('rejects incompatible topology and invalid host windows before allocating outputs', () => {
    const device = createNullWebGPUDevice();
    const timestamps = makeVector(device, 'uint32x2', [2, 1]);
    const validity = makeVector(device, 'uint32', [1, 2]);
    const allocation = vi.spyOn(device, 'createBuffer');
    const props = {
      device,
      timestamps,
      unit: 'microsecond' as const,
      window: {start: 0, end: 1, unit: 'microsecond' as const}
    };
    expect(() => new GPUKeplerTimeSelection({...props, validity})).toThrow(/chunks must align/);
    expect(() => new GPUKeplerTimeSelection({...props, endValidity: validity})).toThrow(
      /requires interval/
    );
    expect(
      () =>
        new GPUKeplerTimeSelection({
          ...props,
          window: {start: 0, end: Number.MAX_SAFE_INTEGER + 1, unit: 'microsecond'}
        })
    ).toThrow(/safe integer/);
    expect(allocation).not.toHaveBeenCalled();
    allocation.mockRestore();
    timestamps.destroy();
    validity.destroy();
    device.destroy();
  });
});

function makeVector<Format extends 'uint32' | 'uint32x2'>(
  device: Device,
  format: Format,
  lengths: number[]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name: 'time-input',
    format,
    data: lengths.map(
      length =>
        new GPUData({
          format,
          length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            byteLength: Math.max(8, length * (format === 'uint32x2' ? 8 : 4)),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
}
