// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {ValueCubeRig} from '../value-cube/value-cube-rig';

// Independent NumPy float32 upload, then Python integer products at scales 65536 and 1:
// f32(1e14)=100000000376832; f32(7e13)=70000001941504.
// Exact scaled totals: 6553600024696061952 and 13762560381715218432.
// The first row clamps to 4611685743549480960; the second total wraps to
// -4684183691994333184. Preserve these existing words, and expose loss separately.
const LEGACY_SUMS = [
  4611685743549480960n,
  -4684183691994333184n,
  -4611685743549480960n,
  4684183691994333184n
];
const EXACT_SMALL_SCALE_SUMS = [
  100000000376832n,
  210000005824512n,
  -100000000376832n,
  -210000005824512n
];
const FLOAT_SUMS = [100000000376832, 210000010018816, -100000000376832, -210000010018816];
const MEANS = [100000000376832, 70000001941504, -100000000376832, -70000001941504];
const VALUES = Float32Array.from([1e14, 7e13, -1e14, -7e13, 7e13, -7e13, 7e13, -7e13, 3e38]);
const KEYS = Uint32Array.from([0, 1, 2, 3, 1, 3, 1, 3, 0]);
const MASK = Uint32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 0]);

for (const sumScale of [65536, 1]) {
  for (const capacity of [4, 257]) {
    for (const scratchTileSize of [undefined, 4]) {
      it(`DGGS-102 groups flag clamping and wrap across tiles (scale=${sumScale}, capacity=${capacity}, tile=${scratchTileSize})`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const rig = new ValueCubeRig(device);
        const sums = rig.output('uint32x2', capacity);
        const sumValues = rig.output('float32', capacity);
        const means = rig.output('float32', capacity);
        const sumOverflow = rig.output('uint32', 1);
        const overflow = rig.output('uint32', 1);
        const keys = rig.output('uint32', capacity);
        const counts = rig.output('uint32', capacity);
        const count = rig.output('uint32', 1);
        const requiredCount = rig.output('uint32', 1);
        try {
          rig.run(
            new GPUGroupStatistics({
              keys: rig.input(KEYS, 'uint32'),
              mask: rig.input(MASK, 'uint32'),
              sumScale,
              scratchTileSize,
              columns: [
                {
                  values: rig.input(VALUES, 'float32'),
                  statistics: ['sum', 'mean'],
                  output: {sums, sumValues, means, status: {sumOverflow}}
                }
              ],
              output: {
                keys,
                counts,
                status: {count, overflow, requiredCount}
              }
            })
          );
          const integers = (await rig.readKeys(sums)).map(value => BigInt.asIntN(64, value));
          expect(integers).toEqual([
            ...(sumScale === 65536 ? LEGACY_SUMS : EXACT_SMALL_SCALE_SUMS),
            ...new Array<bigint>(capacity - 4).fill(0n)
          ]);
          expect(await rig.readFloat(sumValues)).toEqual([
            ...FLOAT_SUMS,
            ...new Array<number>(capacity - 4).fill(0)
          ]);
          expect(await rig.readFloat(means)).toEqual([
            ...MEANS,
            ...new Array<number>(capacity - 4).fill(NaN)
          ]);
          expect(await rig.readUint(sumOverflow)).toEqual([Number(sumScale === 65536)]);
          expect(await rig.readUint(overflow)).toEqual([0]);
          expect(await rig.readUint(keys)).toEqual([
            0,
            1,
            2,
            3,
            ...new Array<number>(capacity - 4).fill(0xffffffff)
          ]);
          expect(await rig.readUint(counts)).toEqual([
            1,
            3,
            1,
            3,
            ...new Array<number>(capacity - 4).fill(0)
          ]);
          expect(await rig.readUint(count)).toEqual([4]);
          expect(await rig.readUint(requiredCount)).toEqual([4]);
        } finally {
          rig.destroy();
        }
      });
    }
  }
  it(`DGGS-102 joins repeat matches and preserve independent float reductions (scale=${sumScale})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new ValueCubeRig(device);
    const sums = rig.output('uint32x2', 6);
    const sumValues = rig.output('float32', 6);
    const means = rig.output('float32', 6);
    const sumOverflow = rig.output('uint32', 1);
    try {
      const values = rig.input(VALUES, 'float32');
      rig.run(
        new GPUKeyLookup({
          leftKeys: rig.input(Uint32Array.from([0, 1, 2, 3, 0, 9]), 'uint32'),
          rightKeys: rig.input(KEYS, 'uint32'),
          rightMask: rig.input(MASK, 'uint32'),
          sumScale,
          aggregates: [
            {operation: 'sum', column: values, output: sumValues, sums, status: {sumOverflow}},
            {operation: 'mean', column: values, output: means}
          ],
          output: {}
        })
      );
      const expected = sumScale === 65536 ? LEGACY_SUMS : EXACT_SMALL_SCALE_SUMS;
      expect((await rig.readKeys(sums)).map(value => BigInt.asIntN(64, value))).toEqual([
        ...expected,
        expected[0],
        0n
      ]);
      expect(await rig.readFloat(sumValues)).toEqual([...FLOAT_SUMS, FLOAT_SUMS[0], NaN]);
      expect(await rig.readFloat(means)).toEqual([...MEANS, MEANS[0], NaN]);
      expect(await rig.readUint(sumOverflow)).toEqual([Number(sumScale === 65536)]);
    } finally {
      rig.destroy();
    }
  });
}

for (const {name, values, sum, sumValue, mean, loss} of [
  {
    name: 'wrap-only',
    values: [2 ** 61, 2 ** 61, 2 ** 61, 2 ** 61],
    sum: -9223372036854775808n,
    sumValue: 2 ** 63,
    mean: 2 ** 61,
    loss: 1
  },
  {
    name: 'final-cancellation',
    values: [2 ** 61, 2 ** 61, 2 ** 61, 2 ** 61, -(2 ** 61)],
    sum: 6917529027641081856n,
    sumValue: 6917529027641081856,
    mean: 1383505860503797800,
    loss: 0
  },
  {
    name: 'clamp-only',
    values: [1e19],
    sum: 4611685743549480960n,
    sumValue: 9999999980506447872,
    mean: 9999999980506447872,
    loss: 1
  }
]) {
  // Closed form: four 2^61 terms total 2^63, appending -2^61 leaves 3*2^61 in range.
  // No row clamps in those two cases. The last fixture has only per-row loss at scale 1.
  // Cancellation mean was independently rounded with NumPy float32(3*2**61/5).
  for (const capacity of [4, 257]) {
    for (const scratchTileSize of [undefined, 2]) {
      it(`B5 groups isolate ${name} across tiles and clear every tail row (capacity=${capacity}, tile=${scratchTileSize})`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const rig = new ValueCubeRig(device);
        const sums = rig.output('uint32x2', capacity);
        const sumValues = rig.output('float32', capacity);
        const means = rig.output('float32', capacity);
        const sumOverflow = rig.output('uint32', 1);
        const keys = rig.output('uint32', capacity);
        const counts = rig.output('uint32', capacity);
        const count = rig.output('uint32', 1);
        const requiredCount = rig.output('uint32', 1);
        const overflow = rig.output('uint32', 1);
        try {
          rig.run(
            new GPUGroupStatistics({
              keys: rig.input(new Uint32Array(values.length).fill(4), 'uint32'),
              sumScale: 1,
              scratchTileSize,
              columns: [
                {
                  values: rig.input(Float32Array.from(values), 'float32'),
                  statistics: ['sum', 'mean'],
                  output: {sums, sumValues, means, status: {sumOverflow}}
                }
              ],
              output: {keys, counts, status: {count, requiredCount, overflow}}
            })
          );
          expect((await rig.readKeys(sums)).map(value => BigInt.asIntN(64, value))).toEqual([
            sum,
            ...new Array<bigint>(capacity - 1).fill(0n)
          ]);
          expect(await rig.readFloat(sumValues)).toEqual([
            sumValue,
            ...new Array<number>(capacity - 1).fill(0)
          ]);
          expect(await rig.readFloat(means)).toEqual([
            mean,
            ...new Array<number>(capacity - 1).fill(NaN)
          ]);
          expect(await rig.readUint(sumOverflow)).toEqual([loss]);
          expect(await rig.readUint(keys)).toEqual([
            4,
            ...new Array<number>(capacity - 1).fill(0xffffffff)
          ]);
          expect(await rig.readUint(counts)).toEqual([
            values.length,
            ...new Array<number>(capacity - 1).fill(0)
          ]);
          expect(await rig.readUint(count)).toEqual([1]);
          expect(await rig.readUint(requiredCount)).toEqual([1]);
          expect(await rig.readUint(overflow)).toEqual([0]);
        } finally {
          rig.destroy();
        }
      });
    }
  }
  it(`B5 joins isolate ${name} independently of every other loss`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new ValueCubeRig(device);
    const sums = rig.output('uint32x2', 3);
    const sumValues = rig.output('float32', 3);
    const means = rig.output('float32', 3);
    const sumOverflow = rig.output('uint32', 1);
    const matchCounts = rig.output('uint32', 3);
    try {
      const column = rig.input(Float32Array.from(values), 'float32');
      rig.run(
        new GPUKeyLookup({
          leftKeys: rig.input(Uint32Array.of(4, 9, 4), 'uint32'),
          rightKeys: rig.input(new Uint32Array(values.length).fill(4), 'uint32'),
          sumScale: 1,
          aggregates: [
            {operation: 'sum', column, output: sumValues, sums, status: {sumOverflow}},
            {operation: 'mean', column, output: means}
          ],
          output: {matchCounts}
        })
      );
      expect((await rig.readKeys(sums)).map(value => BigInt.asIntN(64, value))).toEqual([
        sum,
        0n,
        sum
      ]);
      expect(await rig.readFloat(sumValues)).toEqual([sumValue, NaN, sumValue]);
      expect(await rig.readFloat(means)).toEqual([mean, NaN, mean]);
      expect(await rig.readUint(sumOverflow)).toEqual([loss]);
      expect(await rig.readUint(matchCounts)).toEqual([values.length, 0, values.length]);
    } finally {
      rig.destroy();
    }
  });
}

it.each([
  {left: [4], mask: [1], loss: 0},
  {left: [4, 9], mask: [1, 1], loss: 1},
  {left: [4, 9], mask: [1, 0], loss: 0}
])('B3 join numeric loss is scoped to retained keys $left with mask $mask', async ({
  left,
  mask,
  loss
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new ValueCubeRig(device);
  const sums = rig.output('uint32x2', left.length);
  const sumValues = rig.output('float32', left.length);
  const sumOverflow = rig.output('uint32', 1);
  try {
    rig.run(
      new GPUKeyLookup({
        leftKeys: rig.input(Uint32Array.from(left), 'uint32'),
        leftMask: rig.input(Uint32Array.from(mask), 'uint32'),
        rightKeys: rig.input(Uint32Array.of(4, 9), 'uint32'),
        sumScale: 65536,
        aggregates: [
          {
            operation: 'sum',
            column: rig.input(Float32Array.of(2, 2 ** 46), 'float32'),
            output: sumValues,
            sums,
            status: {sumOverflow}
          }
        ],
        output: {}
      })
    );
    expect((await rig.readKeys(sums)).map(value => BigInt.asIntN(64, value))).toEqual(
      left.length === 1 ? [131072n] : [131072n, mask[1] ? 4611685743549480960n : 0n]
    );
    expect(await rig.readFloat(sumValues)).toEqual(
      left.length === 1 ? [2] : [2, mask[1] ? 2 ** 46 : NaN]
    );
    expect(await rig.readUint(sumOverflow)).toEqual([loss]);
  } finally {
    rig.destroy();
  }
});
