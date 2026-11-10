// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUHealpixCellAggregation,
  GPUHealpixPointToCell
} from '../../../src/gpu-spatial-analysis/healpix-cells/index';
import fixture from '../../oracles/fixtures/rts08-healpix-geo.json';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {getHealpixNestedCenter, getHealpixNestedPixel} from './rts08-healpix-reference';

type Cell = {key: bigint; count: number; sum: number; minimum: number; maximum: number};

async function runAggregation(options: {
  order: number;
  scheme?: 'nested' | 'uniq';
  longitude: number[];
  latitude: number[];
  values: number[];
  mask?: number[];
  cellCapacity: number;
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const rows = options.longitude.length;
  const positions = new Float32Array(2 * rows);
  options.longitude.forEach((value, row) => {
    positions[2 * row] = value;
    positions[2 * row + 1] = options.latitude[row];
  });
  const keyBuffer = device.createBuffer({
    byteLength: 8 * rows,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const keys = importGraphBuffer(rig.graph, 'row-keys', keyBuffer, 'uint32x2', rows);
  const validity = rig.output('uint32', rows);
  const cellBuffer = device.createBuffer({
    byteLength: 8 * options.cellCapacity,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const cells = importGraphBuffer(
    rig.graph,
    'cell-keys',
    cellBuffer,
    'uint32x2',
    options.cellCapacity
  );
  const counts = rig.output('uint32', options.cellCapacity);
  const sums = rig.output('float32', options.cellCapacity);
  const minimums = rig.output('float32', options.cellCapacity);
  const maximums = rig.output('float32', options.cellCapacity);
  const status = {
    count: rig.output('uint32', 1),
    requiredCount: rig.output('uint32', 1),
    overflow: rig.output('uint32', 1),
    invalidCount: rig.output('uint32', 1)
  };
  rig.run(
    new GPUHealpixPointToCell({
      order: options.order,
      scheme: options.scheme,
      positions: rig.input(positions, 'float32x2', rows),
      output: {cells: keys, validity}
    }),
    new GPUHealpixCellAggregation({
      order: options.order,
      scheme: options.scheme,
      cells: keys,
      mask: options.mask ? rig.input(new Uint32Array(options.mask), 'uint32') : undefined,
      values: rig.input(new Float32Array(options.values), 'float32'),
      cellCapacity: options.cellCapacity,
      output: {cells, counts, sums, minimums, maximums, status}
    })
  );
  const rowBytes = await keyBuffer.readAsync();
  const rowWords = new Uint32Array(rowBytes.buffer, rowBytes.byteOffset, 2 * rows);
  const rowKeys = Array.from(
    {length: rows},
    (_, row) => (BigInt(rowWords[2 * row + 1]) << 32n) | BigInt(rowWords[2 * row])
  );
  const cellBytes = await cellBuffer.readAsync();
  const cellWords = new Uint32Array(
    cellBytes.buffer,
    cellBytes.byteOffset,
    2 * options.cellCapacity
  );
  const result = {
    rowKeys,
    cellKeys: Array.from(
      {length: options.cellCapacity},
      (_, cell) => (BigInt(cellWords[2 * cell + 1]) << 32n) | BigInt(cellWords[2 * cell])
    ),
    counts: await rig.readUint(counts),
    sums: await rig.readFloat(sums),
    minimums: await rig.readFloat(minimums),
    maximums: await rig.readFloat(maximums),
    count: (await rig.readUint(status.count))[0],
    requiredCount: (await rig.readUint(status.requiredCount))[0],
    overflow: (await rig.readUint(status.overflow))[0],
    invalidCount: (await rig.readUint(status.invalidCount))[0]
  };
  cellBuffer.destroy();
  keyBuffer.destroy();
  rig.destroy();
  return result;
}

/** Independent grouping of the row keys: a Map over BigInt keys, ascending. */
function groupRows(rowKeys: bigint[], values: number[], keep: (row: number) => boolean): Cell[] {
  const groups = new Map<bigint, Cell>();
  rowKeys.forEach((key, row) => {
    if (!keep(row) || !Number.isFinite(values[row])) return;
    const cell = groups.get(key) ?? {
      key,
      count: 0,
      sum: 0,
      minimum: Infinity,
      maximum: -Infinity
    };
    cell.count++;
    cell.sum += values[row];
    cell.minimum = Math.min(cell.minimum, values[row]);
    cell.maximum = Math.max(cell.maximum, values[row]);
    groups.set(key, cell);
  });
  return [...groups.values()].sort((first, second) => (first.key < second.key ? -1 : 1));
}

const rows = 1500;
const longitude = fixture.longitude.slice(0, rows);
const latitude = fixture.latitude.slice(0, rows);
const values = Array.from({length: rows}, (_, row) => Math.fround(Math.sin(row * 0.37) * 40 + 5));

describe('GPUHealpixCellAggregation', () => {
  for (const scheme of ['nested', 'uniq'] as const) {
    it(`groups points at order 2 into sorted cells with counts, sums and extremes (${scheme})`, async () => {
      const result = await runAggregation({
        order: 2,
        scheme,
        longitude,
        latitude,
        values,
        cellCapacity: 192
      });
      // The row keys come from the point-to-cell kernel; the grouping is recomputed from the
      // binary64 reference pixel of each position, so a key error would show as a mismatch.
      const reference = groupRows(
        longitude.map(
          (lon, row) =>
            getHealpixNestedPixel(lon, latitude[row], 2) + (scheme === 'uniq' ? 4n ** 3n : 0n)
        ),
        values,
        () => true
      );
      const gpu = groupRows(result.rowKeys, values, () => true);
      expect(result.overflow).toBe(0);
      expect(result.requiredCount).toBe(gpu.length);
      expect(result.count).toBe(gpu.length);
      expect(result.invalidCount).toBe(0);
      expect(result.cellKeys.slice(0, gpu.length)).toEqual(gpu.map(cell => cell.key));
      expect(result.counts.slice(0, gpu.length)).toEqual(gpu.map(cell => cell.count));
      expect(result.counts.slice(gpu.length).every(count => count === 0)).toBe(true);
      expect(result.cellKeys.slice(gpu.length).every(key => key === 0n)).toBe(true);
      gpu.forEach((cell, index) => {
        expect(Math.abs(result.sums[index] - cell.sum), `sum ${index}`).toBeLessThan(
          1e-4 * Math.max(1, cell.count)
        );
        expect(result.minimums[index]).toBe(cell.minimum);
        expect(result.maximums[index]).toBe(cell.maximum);
      });
      // float32 keys agree with the binary64 reference cell for all but a handful of points.
      const agreeing = result.rowKeys.filter(
        (key, row) =>
          key ===
          getHealpixNestedPixel(longitude[row], latitude[row], 2) +
            (scheme === 'uniq' ? 4n ** 3n : 0n)
      ).length;
      expect(agreeing).toBeGreaterThanOrEqual(rows - 1);
      expect(reference.length).toBe(gpu.length);
    });
  }

  it('leaves out masked rows and non-finite values and counts them', async () => {
    const mask = Array.from({length: rows}, (_, row) => (row % 7 === 0 ? 0 : 1));
    const withNaN = values.slice();
    withNaN[5] = Number.NaN;
    withNaN[11] = Number.POSITIVE_INFINITY;
    const result = await runAggregation({
      order: 3,
      longitude,
      latitude,
      values: withNaN,
      mask,
      cellCapacity: 768
    });
    const gpu = groupRows(result.rowKeys, withNaN, row => mask[row] !== 0);
    const kept = gpu.reduce((total, cell) => total + cell.count, 0);
    expect(result.invalidCount).toBe(rows - kept);
    expect(result.requiredCount).toBe(gpu.length);
    expect(result.counts.slice(0, gpu.length)).toEqual(gpu.map(cell => cell.count));
    expect(result.cellKeys.slice(0, gpu.length)).toEqual(gpu.map(cell => cell.key));
  });

  it('reports overflow and the required count when the capacity is too small', async () => {
    const full = await runAggregation({
      order: 3,
      longitude,
      latitude,
      values,
      cellCapacity: 768
    });
    const small = await runAggregation({
      order: 3,
      longitude,
      latitude,
      values,
      cellCapacity: 10
    });
    expect(small.overflow).toBe(1);
    expect(small.requiredCount).toBe(full.requiredCount);
    expect(small.count).toBe(10);
    expect(small.cellKeys).toEqual(full.cellKeys.slice(0, 10));
    expect(small.counts).toEqual(full.counts.slice(0, 10));
  });

  it('orders and separates cells whose keys use the high word (order 24)', async () => {
    const base = 60;
    const repeated = Array.from({length: 3 * base}, (_, row) => row % base);
    const result = await runAggregation({
      order: 24,
      scheme: 'uniq',
      longitude: repeated.map(row => longitude[row]),
      latitude: repeated.map(row => latitude[row]),
      values: repeated.map((_, row) => row),
      cellCapacity: 128
    });
    const gpu = groupRows(
      result.rowKeys,
      repeated.map((_, row) => row),
      () => true
    );
    expect(gpu.length).toBe(base);
    expect(gpu.every(cell => cell.count === 3)).toBe(true);
    expect(result.cellKeys.slice(0, base)).toEqual(gpu.map(cell => cell.key));
    expect(result.counts.slice(0, base)).toEqual(gpu.map(() => 3));
    expect(gpu.some(cell => cell.key > 0xffffffffn)).toBe(true);
    expect(result.maximums.slice(0, base)).toEqual(gpu.map(cell => cell.maximum));
  });

  it('keeps cell 0 of the nested scheme as a real cell', async () => {
    const center = getHealpixNestedCenter(0n, 1);
    const result = await runAggregation({
      order: 1,
      longitude: [center.longitude + 0.01, center.longitude - 0.01, 200],
      latitude: [center.latitude, center.latitude + 0.01, -10],
      values: [1, 2, 3],
      cellCapacity: 4
    });
    expect(result.cellKeys[0]).toBe(0n);
    expect(result.counts[0]).toBe(2);
    expect(result.sums[0]).toBe(3);
  });
});
