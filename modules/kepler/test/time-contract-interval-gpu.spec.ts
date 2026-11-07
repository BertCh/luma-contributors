// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {
  classifyIntervalOverlap,
  splitInt64Words,
  type IntervalBoundaryClosure
} from '../../../examples/deck/kepler-compute/src/time-contract-interval';
import {
  ContractIntervalSelection,
  makeContractIntervalTimeSelection
} from '../../../examples/deck/kepler-compute/src/time-contract-interval-gpu';

function makeVector<Format extends 'uint32' | 'uint32x2'>(
  device: Device,
  format: Format,
  chunks: readonly (readonly number[])[],
  width: number
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name: 'test-vector',
    format,
    ownsData: true,
    data: chunks.map(
      chunk =>
        new GPUData({
          format,
          length: chunk.length / width,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data: Uint32Array.from(chunk.length ? chunk : [0]),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  }) as GPUVector<Format>;
}

const VALUES = [
  -(2n ** 63n),
  -5_000_000_000n,
  -1n,
  0n,
  1n,
  4n,
  5n,
  6n,
  10n,
  2n ** 32n,
  2n ** 40n,
  2n ** 63n - 1n
];

describe('time contract interval WGSL on a GPU', () => {
  it('matches the BigInt oracle for all 16 closure combinations, nulls and open ends', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows: {start: bigint | null; end: bigint | null}[] = [];
    for (const start of [...VALUES, null])
      for (const end of [...VALUES, null]) rows.push({start, end});
    const startWords = rows.flatMap(row => splitInt64Words(row.start ?? 0n));
    const endWords = rows.flatMap(row => splitInt64Words(row.end ?? 0n));
    const startValidity = rows.map(row => (row.start === null ? 0 : 1));
    const endValidity = rows.map(row => (row.end === null ? 0 : 1));
    // Split into three uneven chunks to check chunk topology.
    const cut = [37, 101, rows.length];
    const split = <T>(values: readonly T[], width: number): T[][] => {
      const chunks: T[][] = [];
      let from = 0;
      for (const to of cut) {
        chunks.push(values.slice(from * width, to * width));
        from = to;
      }
      return chunks;
    };
    const windows = [
      {start: 0n, end: 5n},
      {start: 5n, end: 5n},
      {start: 4n, end: 6n},
      {start: -(2n ** 63n), end: 2n ** 63n - 1n}
    ];
    for (let combination = 0; combination < 16; combination++) {
      const closure: IntervalBoundaryClosure = {
        rowStartClosed: Boolean(combination & 1),
        rowEndClosed: Boolean(combination & 2),
        windowStartClosed: Boolean(combination & 4),
        windowEndClosed: Boolean(combination & 8)
      };
      const start = makeVector(device, 'uint32x2', split(startWords, 2), 2);
      const end = makeVector(device, 'uint32x2', split(endWords, 2), 2);
      const startValid = makeVector(device, 'uint32', split(startValidity, 1), 1);
      const endValid = makeVector(device, 'uint32', split(endValidity, 1), 1);
      const selection = new ContractIntervalSelection({
        device,
        start,
        end,
        startValidity: startValid,
        endValidity: endValid,
        window: windows[0],
        closure
      });
      try {
        for (const window of windows) {
          selection.setWindow(window);
          const encoder = device.createCommandEncoder();
          selection.encode(encoder);
          device.submit(encoder.finish());
          const mask = await selection.readMask();
          rows.forEach((row, index) => {
            const expected = classifyIntervalOverlap(row, window, closure) === 'overlap' ? 1 : 0;
            expect(
              mask[index],
              `closure ${combination} window ${window.start}..${window.end} row ${row.start}..${row.end}`
            ).toBe(expected);
          });
        }
      } finally {
        selection.destroy();
        start.destroy();
        end.destroy();
        startValid.destroy();
        endValid.destroy();
      }
    }
  });

  it('adapts to the TimeRuntime selection surface: window, reversed window and counts', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const starts = [0n, 10n, 20n, 30n];
    const ends = [5n, 15n, 25n, 35n];
    const start = makeVector(
      device,
      'uint32x2',
      [starts.flatMap(value => splitInt64Words(value))],
      2
    );
    const end = makeVector(device, 'uint32x2', [ends.flatMap(value => splitInt64Words(value))], 2);
    const selection = makeContractIntervalTimeSelection({
      device,
      start,
      end,
      window: {start: 0n, end: 0n}
    });
    try {
      for (const [window, expected] of [
        [{start: 12, end: 22}, 2],
        [{start: 30, end: 20}, 0],
        [{start: -(2n ** 63n), end: 2n ** 63n - 1n}, 4]
      ] as const) {
        selection.setWindow({...window, unit: 'millisecond'});
        const encoder = device.createCommandEncoder();
        selection.encode(encoder);
        device.submit(encoder.finish());
        const status = await selection.readStatus();
        expect(status.selectedCount).toBe(expected);
        expect(status.stale).toBe(false);
      }
    } finally {
      selection.destroy();
      start.destroy();
      end.destroy();
    }
  });
});
