// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUHealpixPointToCell} from '../../../src/gpu-spatial-analysis/healpix-cells/index';
import fixture from '../../oracles/fixtures/rts08-healpix-geo.json';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {getAngleDegrees, getHealpixNestedCenter} from './rts08-healpix-reference';

const getWidth = (depth: number) => 58.6 / 2 ** depth;
const isVertexPoint = (row: number) =>
  fixture.latitude[row] === 0 && fixture.longitude[row] % 45 === 0;

async function runKeys(
  order: number,
  options: {
    scheme?: 'nested' | 'uniq';
    longitude?: number[];
    latitude?: number[];
    mask?: number[];
  } = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const longitude = options.longitude ?? fixture.longitude;
  const latitude = options.latitude ?? fixture.latitude;
  const rows = longitude.length;
  const positions = new Float32Array(2 * rows);
  longitude.forEach((value, row) => {
    positions[2 * row] = value;
    positions[2 * row + 1] = latitude[row];
  });
  const cellBuffer = device.createBuffer({
    byteLength: 8 * rows,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const cells = importGraphBuffer(rig.graph, 'healpix-cells', cellBuffer, 'uint32x2', rows);
  const validity = rig.output('uint32', rows);
  rig.run(
    new GPUHealpixPointToCell({
      order,
      scheme: options.scheme,
      positions: rig.input(positions, 'float32x2', rows),
      mask: options.mask ? rig.input(new Uint32Array(options.mask), 'uint32') : undefined,
      output: {cells, validity}
    })
  );
  const bytes = await cellBuffer.readAsync();
  const words = Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, 2 * rows));
  cellBuffer.destroy();
  const flags = await rig.readUint(validity);
  rig.destroy();
  const keys = Array.from(
    {length: rows},
    (_, row) => (BigInt(words[2 * row + 1]) << 32n) | BigInt(words[2 * row])
  );
  return {keys, flags};
}

/** Rows below this index are random points; the rest are probes on or near cell vertices. */
const RANDOM_ROWS = 3000;

/**
 * Largest share of random points whose float32 key differs from healpix-geo, by order. Measured on
 * the fixture (3000 points): none through order 12, 0.5% at 14, 1.8% at 16, 6.8% at 18 and 23% at
 * 20; the limits leave room for the sampling noise of 3000 points.
 */
const MISMATCH_LIMIT: Record<number, number> = {
  0: 0,
  1: 0,
  2: 0,
  4: 0,
  6: 0,
  8: 0,
  10: 0,
  12: 0,
  14: 0.01,
  16: 0.03,
  18: 0.1,
  20: 0.3
};

function getKeyDistanceWidths(key: bigint, referenceHex: string, depth: number): number {
  return (
    getAngleDegrees(
      getHealpixNestedCenter(key, depth),
      getHealpixNestedCenter(BigInt(`0x${referenceHex}`), depth)
    ) / getWidth(depth)
  );
}

describe('GPUHealpixPointToCell against healpix-geo', () => {
  for (const [depthText, limit] of Object.entries(MISMATCH_LIMIT)) {
    const depth = Number(depthText);
    it(`order ${depth}: at most ${limit * 100}% of random points differ, each by an adjacent cell`, async () => {
      const {keys, flags} = await runKeys(depth);
      const expected = fixture.nestedHex[String(depth) as keyof typeof fixture.nestedHex];
      let mismatches = 0;
      keys.forEach((key, row) => {
        expect(flags[row]).toBe(1);
        if (isVertexPoint(row) || key.toString(16) === expected[row]) return;
        // A neighbour shares an edge or a corner: its center is at most about 1.6 widths away
        // (a face corner joins three cells instead of four).
        expect(getKeyDistanceWidths(key, expected[row], depth), `row ${row}`).toBeLessThan(2);
        if (row < RANDOM_ROWS) mismatches++;
      });
      expect(mismatches / RANDOM_ROWS).toBeLessThanOrEqual(limit);
    });
  }

  for (const depth of [24, 29]) {
    it(`order ${depth}: keys the cell of the position within float32 quantization`, async () => {
      const {keys} = await runKeys(depth);
      keys.forEach((key, row) => {
        if (row >= RANDOM_ROWS) return;
        const center = getHealpixNestedCenter(key, depth);
        const point = {longitude: fixture.longitude[row], latitude: fixture.latitude[row]};
        expect(getAngleDegrees(center, point), `row ${row}`).toBeLessThan(1e-3);
      });
    });
  }

  it('uniq adds 4^(order + 1) and is never zero, where nested cell 0 is zero', async () => {
    const order = 2;
    const center = getHealpixNestedCenter(0n, order);
    const longitude = [center.longitude, ...fixture.longitude.slice(0, 20)];
    const latitude = [center.latitude, ...fixture.latitude.slice(0, 20)];
    const nested = await runKeys(order, {longitude, latitude});
    const uniq = await runKeys(order, {longitude, latitude, scheme: 'uniq'});
    expect(nested.keys[0]).toBe(0n);
    expect(nested.flags[0]).toBe(1);
    expect(uniq.keys[0]).toBe(4n ** BigInt(order + 1));
    nested.keys.forEach((key, row) => {
      expect(uniq.keys[row]).toBe(key + 4n ** BigInt(order + 1));
    });
  });

  it('gives masked and non-finite rows the zero key and a zero validity flag', async () => {
    const longitude = [10, Number.NaN, 20, 30];
    const latitude = [10, 10, Number.POSITIVE_INFINITY, 40];
    const result = await runKeys(5, {longitude, latitude, mask: [1, 1, 1, 0], scheme: 'uniq'});
    expect(result.flags).toEqual([1, 0, 0, 0]);
    expect(result.keys[0]).not.toBe(0n);
    expect(result.keys.slice(1)).toEqual([0n, 0n, 0n]);
  });

  it('rejects an order above 29', () => {
    expect(
      () =>
        new GPUHealpixPointToCell({
          order: 30,
          positions: {length: 1} as never,
          output: {cells: {} as never}
        })
    ).toThrow(/healpix-point-to-cell order must be an integer in \[0, 29\]/);
  });
});
