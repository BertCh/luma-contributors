// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  decodeGPURegionStatistics,
  getGPURegionStatisticsSummaryLength,
  GPURegionStatisticsReadback
} from '../../../src/gpu-spatial-analysis/region-statistics/index';

it('R6: a failed copy releases its reservation so a valid immediate retry succeeds', () => {
  const device = createNullWebGPUDevice();
  const readback = new GPURegionStatisticsReadback(device, {slotCount: 1});
  const short = device.createBuffer({byteLength: 28, usage: Buffer.COPY_SRC});
  const valid = device.createBuffer({byteLength: 32, usage: Buffer.COPY_SRC});
  const encoder = device.createCommandEncoder();
  try {
    expect(() => readback.encodeRead(encoder, short)).toThrow();
    expect(readback.ring.availableSlotCount).toBe(1);
    expect(readback.encodeRead(encoder, valid)).not.toBeNull();
  } finally {
    readback.destroy();
    short.destroy();
    valid.destroy();
    encoder.destroy();
    device.destroy();
  }
});

it('R11: readback binCount is an integer in [0,65536], including a successful zero-bin header', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const binCount of [-1, 0.5, NaN, Infinity, 65537]) {
      expect
        .soft(() => getGPURegionStatisticsSummaryLength(binCount), String(binCount))
        .toThrow(/binCount/);
      // Do not leave the pre-fix successfully constructed invalid rings allocated during reproduction.
      expect
        .soft(() => {
          const readback = new GPURegionStatisticsReadback(device, {binCount});
          readback.destroy();
        }, String(binCount))
        .toThrow();
    }
    const valid = new GPURegionStatisticsReadback(device, {binCount: 0});
    expect(valid.byteLength).toBe(32);
    valid.destroy();
  } finally {
    device.destroy();
  }
});

it('getGPURegionStatisticsSummaryLength adds the header', () => {
  expect(getGPURegionStatisticsSummaryLength(0)).toBe(8);
  expect(getGPURegionStatisticsSummaryLength(16)).toBe(24);
});

it('decodeGPURegionStatistics round-trips every field', () => {
  const words = new Uint32Array(12);
  const floats = new Float32Array(words.buffer);
  words[0] = 5;
  words[1] = 4;
  floats[2] = 55;
  floats[3] = 13.75;
  floats[4] = -5;
  floats[5] = 30;
  words[6] = 1;
  words[7] = 3;
  words.set([1, 2, 3, 4], 8);
  const result = decodeGPURegionStatistics(words);
  expect(result).toMatchObject({
    selectedCount: 5,
    valueCount: 4,
    sum: 55,
    mean: 13.75,
    minimum: -5,
    maximum: 30,
    histogramOutsideCount: 1,
    selectionTruncated: true,
    regionTruncated: true
  });
  expect(Array.from(result.histogram)).toEqual([1, 2, 3, 4]);
  expect(() => decodeGPURegionStatistics(new Uint8Array(28))).toThrow();
  expect(() => decodeGPURegionStatistics(new Uint8Array(34))).toThrow();
});
