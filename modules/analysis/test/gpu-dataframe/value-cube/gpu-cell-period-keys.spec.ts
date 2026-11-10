// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {gridDisk, latLngToCell} from 'h3-js';
import {describe, expect, it} from 'vitest';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/index';
import {
  GPUCellPeriodKeys,
  getGPUCellKeyFreeBits,
  getGPUCellPeriodKeyPeriodBits,
  packGPUCellPeriodKey,
  unpackGPUCellPeriodKey,
  type GPUCellPeriodKeyFill
} from '../../../src/dataframe-analysis/value-cube/index';
import {
  h3ToBigInt,
  quadbinTileToCell
} from '../../gpu-spatial-analysis/cell-aggregation/cell-aggregation-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import {createValueCubeRandom, getKeyWords, ValueCubeRig} from './value-cube-rig';

const ALL_ONES = 0xffffffffffffffffn;
const PERIOD_COUNT = 365;

/** S2 cell id of a path of `2 * level` bits on a face: face, position, sentinel one, zeros. */
function getS2CellId(face: number, path: bigint, level: number): bigint {
  const shift = BigInt(60 - 2 * level);
  return (BigInt(face) << 61n) | (path << (shift + 1n)) | (1n << shift);
}

type Family = {
  name: string;
  fill: GPUCellPeriodKeyFill;
  /** Cells coarse enough for 9 period bits. */
  coarse: bigint[];
  /** Cells too fine for 9 period bits. */
  fine: bigint[];
};

function createFamilies(): Family[] {
  const random = createValueCubeRandom(3);
  const origin = latLngToCell(41.88, -87.63, 9);
  const h3Coarse = gridDisk(origin, 3).map(h3ToBigInt);
  const h3Fine = gridDisk(latLngToCell(41.88, -87.63, 13), 1).map(h3ToBigInt);
  const quadbinCoarse = Array.from({length: 40}, () =>
    quadbinTileToCell(Math.floor(random() * 4096), Math.floor(random() * 4096), 12)
  );
  const quadbinFine = Array.from({length: 6}, () =>
    quadbinTileToCell(Math.floor(random() * 2 ** 24), Math.floor(random() * 2 ** 24), 24)
  );
  const s2Coarse = Array.from({length: 40}, () =>
    getS2CellId(Math.floor(random() * 6), BigInt(Math.floor(random() * 2 ** 40)), 20)
  );
  const s2Fine = Array.from({length: 6}, () =>
    getS2CellId(Math.floor(random() * 6), BigInt(Math.floor(random() * 2 ** 56)), 28)
  );
  return [
    {name: 'H3', fill: 'ones', coarse: h3Coarse, fine: h3Fine},
    {name: 'Quadbin', fill: 'ones', coarse: quadbinCoarse, fine: quadbinFine},
    {name: 'S2', fill: 'zeros', coarse: s2Coarse, fine: s2Fine}
  ];
}

describe('GPUCellPeriodKeys', () => {
  it('states the free key bits per resolution', () => {
    expect(getGPUCellPeriodKeyPeriodBits(365)).toBe(9);
    expect(getGPUCellPeriodKeyPeriodBits(512)).toBe(9);
    expect(getGPUCellPeriodKeyPeriodBits(513)).toBe(10);
    expect(getGPUCellPeriodKeyPeriodBits(1)).toBe(1);
    // Derived from the key layouts: H3 digits below the resolution are 7, Quadbin footer is ones.
    expect(getGPUCellKeyFreeBits('h3', 9)).toBe(18);
    expect(getGPUCellKeyFreeBits('quadbin', 12)).toBe(28);
    expect(getGPUCellKeyFreeBits('s2', 20)).toBe(20);
    // The real keys agree with the stated counts: the free bits hold the fill pattern.
    const h3 = h3ToBigInt(latLngToCell(41.88, -87.63, 9));
    expect(h3 & ((1n << 18n) - 1n)).toBe((1n << 18n) - 1n);
    // The digit at the resolution is a real digit (0 to 6), never the unused-digit value 7.
    expect((h3 >> 18n) & 7n).not.toBe(7n);
    const quadbin = quadbinTileToCell(5, 9, 12);
    expect(quadbin & ((1n << 28n) - 1n)).toBe((1n << 28n) - 1n);
    const s2 = getS2CellId(2, 12345n, 20);
    expect(s2 & ((1n << 20n) - 1n)).toBe(0n);
    expect((s2 >> 20n) & 1n).toBe(1n);
  });

  for (const family of createFamilies()) {
    it(`packs ${family.name} keys with periods, rejects cells that are too fine, and feeds GPUGroupStatistics`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const random = createValueCubeRandom(17);
      const rowCount = 3000;
      const cellKeys: bigint[] = [];
      const periods: number[] = [];
      const values: number[] = [];
      const mask: number[] = [];
      for (let row = 0; row < rowCount; row++) {
        const kind = random();
        const pool = kind < 0.04 ? family.fine : family.coarse;
        cellKeys.push(kind > 0.99 ? ALL_ONES : pool[Math.floor(random() * pool.length)]);
        periods.push(
          kind > 0.96 && kind <= 0.98
            ? 0xffffffff
            : kind > 0.98 && kind <= 0.99
              ? PERIOD_COUNT + 3
              : // A few cells and periods repeat a lot, so groups have many rows.
                Math.floor(random() * random() * PERIOD_COUNT)
        );
        // Multiples of 1/16 below 2^10: exact in GPUGroupStatistics' 1/65536 fixed point.
        values.push(Math.floor(random() * 16384) / 16 - 400);
        mask.push(random() < 0.9 ? 1 : 0);
      }
      // Independent expectation: group the events by the (cell, period) pair itself.
      const expectedGroups = new Map<string, {cell: bigint; period: number; values: number[]}>();
      let expectedInvalid = 0;
      for (let row = 0; row < rowCount; row++) {
        const cell = cellKeys[row];
        const period = periods[row];
        if (!mask[row] || cell === ALL_ONES || period === 0xffffffff) continue;
        // Packable exactly when the low 9 bits hold the fill; a finer cell whose extra bits happen
        // to equal the fill is still a distinct key that unpacks to itself.
        const lowBits = cell & 511n;
        const isFillPattern = lowBits === (family.fill === 'ones' ? 511n : 0n);
        if (period >= PERIOD_COUNT || !isFillPattern) {
          expectedInvalid++;
          continue;
        }
        const label = `${cell}:${period}`;
        const group = expectedGroups.get(label) ?? {cell, period, values: []};
        group.values.push(values[row]);
        expectedGroups.set(label, group);
      }
      expect(expectedInvalid).toBeGreaterThan(20);
      expect(expectedGroups.size).toBeGreaterThan(200);

      const rig = new ValueCubeRig(device);
      const packed = rig.output('uint32x2', rowCount);
      const invalidCount = rig.output('uint32', 1);
      const capacity = 4096;
      const groupKeys = rig.output('uint32x2', capacity);
      const groupCounts = rig.output('uint32', capacity);
      const columnCounts = rig.output('uint32', capacity);
      const sumValues = rig.output('float32', capacity);
      const means = rig.output('float32', capacity);
      const groupCount = rig.output('uint32', 1);
      const overflow = rig.output('uint32', 1);
      const valuesView = rig.input(Float32Array.from(values), 'float32');
      rig.run(
        new GPUCellPeriodKeys({
          cells: rig.input(getKeyWords(cellKeys), 'uint32x2'),
          periods: rig.input(Uint32Array.from(periods), 'uint32'),
          periodCount: PERIOD_COUNT,
          fill: family.fill,
          mask: rig.input(Uint32Array.from(mask), 'uint32'),
          output: {keys: packed, status: {invalidCount}}
        }),
        new GPUGroupStatistics({
          keys: packed,
          mask: rig.input(Uint32Array.from(mask), 'uint32'),
          columns: [
            {
              values: valuesView,
              statistics: ['count', 'sum', 'mean'],
              output: {counts: columnCounts, sumValues, means}
            }
          ],
          output: {keys: groupKeys, counts: groupCounts, status: {count: groupCount, overflow}}
        })
      );

      // The packing kernel against the CPU packing of the same rows.
      const packedKeys = await rig.readKeys(packed);
      for (let row = 0; row < rowCount; row++) {
        const wanted =
          !mask[row] || cellKeys[row] === ALL_ONES || periods[row] === 0xffffffff
            ? ALL_ONES
            : (packGPUCellPeriodKey(cellKeys[row], periods[row], PERIOD_COUNT, family.fill) ??
              ALL_ONES);
        expect(packedKeys[row], `row ${row}`).toBe(wanted);
      }
      expect((await rig.readUint(invalidCount))[0]).toBe(expectedInvalid);

      // The whole sparse cube against the independent grouping, through unpack.
      const [groups] = await rig.readUint(groupCount);
      expect(groups).toBe(expectedGroups.size);
      expect((await rig.readUint(overflow))[0]).toBe(0);
      const keys = (await rig.readKeys(groupKeys)).slice(0, groups);
      expect([...keys].sort((a, b) => (a < b ? -1 : 1))).toEqual(keys);
      const counts = await rig.readUint(columnCounts);
      const sums = await rig.readFloat(sumValues);
      const meansRead = await rig.readFloat(means);
      keys.forEach((key, index) => {
        const {cellKey, period} = unpackGPUCellPeriodKey(key, PERIOD_COUNT, family.fill);
        const group = expectedGroups.get(`${cellKey}:${period}`);
        expect(group, `group ${cellKey}:${period}`).toBeDefined();
        const total = group!.values.reduce((sum, value) => sum + value, 0);
        expect(counts[index]).toBe(group!.values.length);
        expect(sums[index]).toBeCloseTo(total, 3);
        expect(Math.abs(meansRead[index] - total / group!.values.length)).toBeLessThan(
          1e-4 * (1 + Math.abs(total / group!.values.length))
        );
      });
      rig.destroy();
    });
  }
});
