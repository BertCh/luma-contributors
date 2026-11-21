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

/**
 * DGGS-104: just west of a 90 degree meridian at the belt edge (latitude asin(2/3) = 41.8103149),
 * f32 rounds `0.5 + tt` up to 4.5, so a belt diagonal reached `5 nside`: the key was the far
 * corner of face 4 in the north (order 4: 1194, 56 degrees away) and the invalid face 12 in the
 * south. A point that close to the face corner may take any of the three cells meeting there.
 * Allowed keys are ang2pix_nest (Gorski et al. 2005) in 60-digit mpmath at offsets of
 * {0, +-1e-6, +-1e-4} degrees in longitude and +-{1e-5, 1e-4} in latitude around the corner; that
 * transcription matches the healpix-geo fixture above on 6,035 sampled rows (every row except
 * the exact vertex probe (90, 0)).
 */
const BELT_CORNER_KEYS: Record<number, {north: bigint[]; south: bigint[]}> = {
  4: {north: [170n, 853n, 1279n], south: [1024n, 2218n, 2901n]},
  12: {
    north: [11184810n, 55924053n, 83886079n],
    south: [67108864n, 145402538n, 190141781n]
  }
};

describe('GPUHealpixPointToCell at the belt edge west of a 90 degree meridian', () => {
  const longitude: number[] = [-1.54e-5, -4e-5];
  const latitude: number[] = [41.8103142, -41.8102913];
  // The sweep the review asked for: latitude 41.81028 to 41.81032 in 1e-6 steps, both signs.
  for (let step = 0; step <= 40; step++) {
    for (const sign of [1, -1]) {
      longitude.push(-3e-5);
      latitude.push(sign * (41.81028 + step * 1e-6));
    }
  }
  for (const order of [4, 12]) {
    it(`order ${order}: every key is one of the corner cells`, async () => {
      const {keys, flags} = await runKeys(order, {longitude, latitude});
      keys.forEach((key, row) => {
        expect(flags[row]).toBe(1);
        const allowed = BELT_CORNER_KEYS[order][latitude[row] > 0 ? 'north' : 'south'];
        expect(allowed, `row ${row} (${longitude[row]}, ${latitude[row]}) key ${key}`).toContain(
          key
        );
      });
    });
  }

  it('reduces longitudes beyond 2^24 exactly modulo 360', async () => {
    // Float32 longitudes 1e8, 2^30, -3e9, 1e30 and -3.4e38 (residues 280, 64, 240, 120 and 136
    // from Fraction(float32) % 360); keys from the same mpmath ang2pix_nest.
    const largeLongitude = [1e8, 2 ** 30, -3e9, 1e30, -3.4e38];
    const largeLatitude = [10, 60, -30, 5, -70];
    const expected: Record<number, bigint[]> = {
      4: [1989n, 209n, 2682n, 1374n, 2319n],
      12: [130380280n, 13738371n, 175793766n, 90086953n, 151992142n]
    };
    for (const order of [4, 12]) {
      const {keys, flags} = await runKeys(order, {
        longitude: largeLongitude,
        latitude: largeLatitude
      });
      expect(flags).toEqual([1, 1, 1, 1, 1]);
      expect(keys, `order ${order}`).toEqual(expected[order]);
    }
  });
});
