// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {GPUValueCube} from '../../../src/dataframe-analysis/value-cube/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {computeValueCubeOracle, getExactFloat32Sum, type ValueCubeEvent} from './value-cube-oracle';
import {createValueCubeRandom, ValueCubeRig} from './value-cube-rig';

const INVALID_PERIOD = 0xffffffff;
const OUT_OF_RANGE_PERIOD = 0xfffffffe;

async function runCube(
  events: readonly ValueCubeEvent[],
  cellCount: number,
  periodCount: number,
  options: {withMask: boolean}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new ValueCubeRig(device);
  const binCount = cellCount * periodCount;
  const counts = rig.output('uint32', binCount);
  const sums = rig.output('float32', binCount);
  const means = rig.output('float32', binCount);
  const invalidCount = rig.output('uint32', 1);
  rig.run(
    new GPUValueCube({
      cellCount,
      periodCount,
      cells: rig.input(
        Uint32Array.from(events, event => event.cell),
        'uint32'
      ),
      periods: rig.input(
        Uint32Array.from(events, event => event.period),
        'uint32'
      ),
      values: rig.input(
        Float32Array.from(events, event => event.value),
        'float32'
      ),
      mask: options.withMask
        ? rig.input(
            Uint32Array.from(events, event => (event.selected ? 1 : 0)),
            'uint32'
          )
        : undefined,
      output: {counts, sums, means, status: {invalidCount}}
    })
  );
  const result = {
    counts: await rig.readUint(counts),
    sums: await rig.readFloat(sums),
    means: await rig.readFloat(means),
    invalidCount: (await rig.readUint(invalidCount))[0]
  };
  rig.destroy();
  return result;
}

/** Bit pattern, so that -0, +0 and NaN payloads are compared exactly. */
function bitsOf(values: readonly number[]): number[] {
  const floats = Float32Array.from(values);
  return Array.from(new Uint32Array(floats.buffer));
}

describe('GPUValueCube', () => {
  it('matches the exact big-integer cube bit for bit with masked, invalid and non-finite rows', async () => {
    const cellCount = 37;
    const periodCount = 11;
    const random = createValueCubeRandom(5);
    const events: ValueCubeEvent[] = Array.from({length: 6000}, () => {
      const kind = random();
      let period = Math.floor(random() * periodCount);
      let cell = Math.floor(random() * cellCount);
      let value = Math.fround((random() - 0.3) * 2000);
      if (kind < 0.03) period = INVALID_PERIOD;
      else if (kind < 0.06) period = OUT_OF_RANGE_PERIOD;
      else if (kind < 0.08) cell = cellCount + Math.floor(random() * 5);
      else if (kind < 0.1) value = NaN;
      else if (kind < 0.11) value = Infinity;
      return {cell, period, value, selected: random() > 0.15};
    });
    const expected = computeValueCubeOracle(events, cellCount, periodCount);
    expect(expected.invalidCount).toBeGreaterThan(100);
    expect(new Set(expected.counts).size).toBeGreaterThan(10);
    const actual = await runCube(events, cellCount, periodCount, {withMask: true});
    expect(actual.counts).toEqual(expected.counts);
    expect(bitsOf(actual.sums)).toEqual(bitsOf(expected.sums));
    // The independent integer-ratio oracle and production publication both round once.
    expect(bitsOf(actual.means)).toEqual(bitsOf(expected.means));
    expect(actual.invalidCount).toBe(expected.invalidCount);
    // An empty bin has sum 0 and mean NaN, the missing marker of GPUEmergingHotSpots.
    const emptyBin = expected.counts.indexOf(0);
    if (emptyBin >= 0) {
      expect(actual.sums[emptyBin]).toBe(0);
      expect(actual.means[emptyBin]).toBeNaN();
    }
  });

  it('is exact where float32 accumulation is order dependent', async () => {
    // One bin with huge cancelling terms around a small one: any float32 running sum gives 0 or
    // another wrong value for some order, the exact sum is the small values' total.
    const events: ValueCubeEvent[] = [];
    const smallValues: number[] = [];
    for (let index = 0; index < 300; index++) {
      const small = Math.fround(1 + (index % 7) / 8);
      smallValues.push(small);
      events.push({cell: 0, period: 0, value: 1.0e8, selected: true});
      events.push({cell: 0, period: 0, value: small, selected: true});
      events.push({cell: 0, period: 0, value: -1.0e8, selected: true});
    }
    // Shuffle so the large terms are interleaved arbitrarily.
    const random = createValueCubeRandom(9);
    for (let row = events.length - 1; row > 0; row--) {
      const other = Math.floor(random() * (row + 1));
      [events[row], events[other]] = [events[other], events[row]];
    }
    let floatSum = Math.fround(0);
    for (const event of events) floatSum = Math.fround(floatSum + event.value);
    const exact = getExactFloat32Sum(smallValues);
    expect(exact).toBeGreaterThan(300);
    // The naive float32 sum is wrong, so equality below is a real test of the accumulator.
    expect(floatSum).not.toBe(exact);
    const actual = await runCube(events, 1, 1, {withMask: false});
    expect(actual.sums[0]).toBe(exact);
    expect(actual.counts[0]).toBe(events.length);
    expect(Math.abs(actual.means[0] / (exact / events.length) - 1)).toBeLessThan(4e-7);
  });

  it('review DPC-109: keeps a finite mean when a bin sum is past the float32 range', async () => {
    // Four values of 1e38 in bin 0: the exact sum 4e38 rounds to Infinity in float32, and before
    // the fix the mean was Infinity / 4. The mean of four equal values is the value (closed form):
    // fround(1e38) = 9.999999680285692e37. Bin 1 is an ordinary control; its values must keep
    // their bits in the fixed-point unit, which follows the largest value (about 2^35 here), so
    // 1 and 3 would quantize to 0. 2^120 and 3 * 2^120 are exact: sum 2^122, mean 2^121.
    const events: ValueCubeEvent[] = [
      ...Array.from({length: 4}, () => ({cell: 0, period: 0, value: 1e38, selected: true})),
      {cell: 1, period: 0, value: 2 ** 120, selected: true},
      {cell: 1, period: 0, value: 3 * 2 ** 120, selected: true}
    ];
    const actual = await runCube(events, 2, 1, {withMask: false});
    expect(actual.counts).toEqual([4, 2]);
    expect(actual.sums[0]).toBe(Infinity);
    expect(Number.isFinite(actual.means[0]), `mean ${actual.means[0]}`).toBe(true);
    expect(actual.means[0]).toBe(9.999999680285692e37);
    expect(actual.sums[1]).toBe(2 ** 122);
    expect(actual.means[1]).toBe(2 ** 121);
  });

  it('reproduces the same cube when rows are reordered', async () => {
    const random = createValueCubeRandom(21);
    const events: ValueCubeEvent[] = Array.from({length: 2000}, () => ({
      cell: Math.floor(random() * 5),
      period: Math.floor(random() * 4),
      value: Math.fround((random() - 0.5) * 1e5 * (random() < 0.05 ? 1e4 : 1)),
      selected: true
    }));
    const reversed = [...events].reverse();
    const first = await runCube(events, 5, 4, {withMask: false});
    const second = await runCube(reversed, 5, 4, {withMask: false});
    expect(bitsOf(second.sums)).toEqual(bitsOf(first.sums));
    expect(second.counts).toEqual(first.counts);
  });
});

it('CMP-12 event rows retain Monday and Tuesday occupancy despite equal all-period totals', async () => {
  for (const period of [1, 2]) {
    const result = await runCube(
      [
        {cell: 0, period: 0, value: 1, selected: true},
        {cell: 1, period, value: 1, selected: true}
      ],
      2,
      3,
      {withMask: false}
    );
    expect(result.counts).toEqual(period === 1 ? [1, 0, 0, 0, 1, 0] : [1, 0, 0, 0, 0, 1]);
  }
});

it('W2-03 seven maximum values retain finite cube means and independent infinite sums', async () => {
  const result = await runCube(
    Array.from({length: 7}, () => ({
      cell: 0,
      period: 0,
      value: 3.4028234663852886e38,
      selected: true
    })),
    1,
    2,
    {withMask: false}
  );
  expect(result.counts).toEqual([7, 0]);
  expect(result.means).toEqual([3.4028234663852886e38, NaN]);
  expect(result.sums).toEqual([Infinity, 0]);
});

it('UDF-06 per-bin scale retains two independent occupied singleton bins', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new ValueCubeRig(device);
  try {
    const sums = rig.output('float32', 2);
    const means = rig.output('float32', 2);
    rig.run(
      new GPUValueCube({
        scaleMode: 'per-bin',
        cellCount: 1,
        periodCount: 2,
        cells: rig.input(Uint32Array.of(0, 0), 'uint32'),
        periods: rig.input(Uint32Array.of(0, 1), 'uint32'),
        values: rig.input(Float32Array.of(1, 2 ** 100), 'float32'),
        output: {sums, means}
      })
    );
    expect(await rig.readFloat(sums)).toEqual([1, 2 ** 100]);
    expect(await rig.readFloat(means)).toEqual([1, 2 ** 100]);
  } finally {
    rig.destroy();
  }
});

it('W2-03 production cube publication retains signed exact totals at count 16777217', async () => {
  const {createTransientView} = await import('@luma.gl/graph/gpu-core');
  const {createWGSLKernelNode} = await import(
    '../../../src/analysis-core/kernels/wgsl-kernel-nodes'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new ValueCubeRig(device);
  const means = rig.output('float32', 3);
  const count = 16777217;
  const total = BigInt(count) * 0xffffffn * (1n << 46n);
  const words = [total, -total, 0n].flatMap(value => {
    const unsigned = BigInt.asUintN(96, value);
    return [
      Number(unsigned & 0xffffffffn),
      Number((unsigned >> 32n) & 0xffffffffn),
      Number(unsigned >> 64n)
    ];
  });
  // E=254, headroom=25 => unit exponent 58. maxFloat32=(2^24-1)*2^104.
  const cube = new GPUValueCube({
    cellCount: 1,
    periodCount: 3,
    cells: createTransientView(rig.graph, 'synthetic-cells', 'uint32', count),
    periods: createTransientView(rig.graph, 'synthetic-periods', 'uint32', count),
    values: createTransientView(rig.graph, 'synthetic-values', 'float32', count),
    output: {means}
  });
  const finish = cube.getCommandNodes(rig.graph).find(node => node.id === 'value-cube-means')!;
  try {
    for (const [resourceIndex, data] of [
      [0, Uint32Array.of(count, count, 0)],
      [2, Uint32Array.from(words)],
      [3, Uint32Array.of(254)]
    ] as const) {
      const resource = finish.resources![resourceIndex];
      if (!('buffer' in resource)) throw new Error('publication buffer required');
      const source = rig.input(data, 'uint32');
      rig.graph.add(
        createWGSLKernelNode(rig.graph, {
          id: `inject-${resourceIndex}`,
          operation: 'mean-regression',
          variant: 'inject',
          invocationCount: data.length,
          bindings: [
            {name: 'source', view: source, type: 'u32', access: 'read'},
            {
              name: 'destination',
              view: resource.buffer as typeof source,
              type: 'u32',
              access: 'read_write'
            }
          ],
          body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
        })
      );
    }
    rig.graph.add(finish);
    rig.run();
    expect(await rig.readFloat(means)).toEqual([
      3.4028234663852886e38,
      -3.4028234663852886e38,
      NaN
    ]);
  } finally {
    rig.destroy();
  }
});

it('W3-01 cube subnormal mean has independently rounded word 0x00400001', async () => {
  const result = await runCube(
    [2 ** -126, -(2 ** -126), 12582914 * 2 ** -149].map(value => ({
      cell: 0,
      period: 0,
      value,
      selected: true
    })),
    1,
    1,
    {withMask: false}
  );
  // 12582914 / 3 = 4194304 + 2/3 subnormal units.
  expect(bitsOf(result.means)).toEqual([0x00400001]);
});
