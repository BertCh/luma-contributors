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
    // WGSL division is not correctly rounded (up to 2.5 ULP), so a mean may differ from the
    // correctly rounded quotient by a few units in the last place; NaN must match exactly.
    const actualMeanBits = bitsOf(actual.means);
    const expectedMeanBits = bitsOf(expected.means);
    for (let bin = 0; bin < expected.means.length; bin++) {
      if (Number.isNaN(expected.means[bin])) {
        expect(actual.means[bin], `mean ${bin}`).toBeNaN();
      } else {
        expect(
          Math.abs(actualMeanBits[bin] - expectedMeanBits[bin]),
          `mean ${bin}`
        ).toBeLessThanOrEqual(3);
      }
    }
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
