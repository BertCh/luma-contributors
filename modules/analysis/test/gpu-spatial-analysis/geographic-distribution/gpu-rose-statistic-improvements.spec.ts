// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURoseStatistic} from '../../../src/gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';

it.each([
  2 ** 126,
  2 ** -126
])('Rose moments normalize finite weights at scale %s without squared overflow or underflow', async weight => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'rose-weight-scale');
  try {
    const summary = fixture.output('summary', 'float32', 5);
    const bins = fixture.output('bins', 'float32', 4);
    fixture.graph.add(
      new GPURoseStatistic({
        starts: fixture.input('starts', new Float32Array(4), 'float32x2', 2),
        ends: fixture.input('ends', new Float32Array([1, 0, 0, 1]), 'float32x2', 2),
        weights: fixture.input('weights', new Float32Array([weight, weight]), 'float32', 2),
        circular: {
          mode: 'directional',
          angleUnit: 'radians',
          origin: 'positive-x-counter-clockwise',
          binCount: 4
        },
        output: {summary: summary.view, bins: bins.view}
      })
    );
    fixture.run();
    const values = await fixture.readFloat32(summary, 5);
    expect(values[0]).toBe(2);
    expect(values[1]).toBeCloseTo(Math.PI / 4, 6);
    expect(values[2]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(values[3]).toBeCloseTo(1 - Math.SQRT1_2, 6);
    expect(values[4]).toBeCloseTo(1, 6);
    expect(await fixture.readFloat32(bins, 4)).toEqual([weight, weight, 0, 0]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  'directional',
  'axial'
] as const)('degree bearings preserve %s moments and assign compass bins', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'rose-bearing-conventions');
  try {
    const starts = fixture.input('starts', new Float32Array(8), 'float32x2', 4);
    const ends = fixture.input(
      'ends',
      new Float32Array([1, 0, 0, 1, -1, 0, 0, -1]),
      'float32x2',
      4
    );
    const weights = fixture.input('weights', new Float32Array([4, 2, 1, 1]), 'float32', 4);
    const results = ['positive-x-counter-clockwise', 'positive-y-clockwise'].map(
      (origin, convention) => {
        const summary = fixture.output(`summary-${convention}`, 'float32', 5);
        const bins = fixture.output(`bins-${convention}`, 'float32', 4);
        fixture.graph.add(
          new GPURoseStatistic({
            id: `rose-${convention}`,
            starts,
            ends,
            weights,
            circular: {
              mode,
              angleUnit: 'degrees',
              origin: origin as 'positive-x-counter-clockwise' | 'positive-y-clockwise',
              binCount: 4
            },
            output: {summary: summary.view, bins: bins.view}
          })
        );
        return {summary, bins};
      }
    );
    fixture.run();
    for (const [convention, output] of results.entries()) {
      const summary = await fixture.readFloat32(output.summary, 5);
      expect(summary[0]).toBe(4);
      const expectedMean =
        mode === 'axial'
          ? convention === 0
            ? 0
            : 90
          : convention === 0
            ? (Math.atan2(1, 3) * 180) / Math.PI
            : 90 - (Math.atan2(1, 3) * 180) / Math.PI;
      // An axial mean infinitesimally below zero can wrap to 180 under f32 sin(pi) error.
      const period = mode === 'axial' ? 180 : 360;
      expect(Number.isFinite(summary[1])).toBe(true);
      expect(summary[1]).toBeGreaterThanOrEqual(0);
      expect(summary[1]).toBeLessThan(period);
      const distance = (((summary[1] - expectedMean) % period) + period) % period;
      expect(Math.min(distance, period - distance)).toBeLessThan(0.0001);
      expect(summary[2]).toBeCloseTo(mode === 'axial' ? 0.25 : Math.sqrt(10) / 8, 5);
      expect(summary[3]).toBeCloseTo(1 - summary[2], 6);
      expect(summary[4]).toBeCloseTo(0, 5);
      expect(await fixture.readFloat32(output.bins, 4)).toEqual(
        mode === 'axial'
          ? convention === 0
            ? [5, 0, 3, 0]
            : [3, 0, 5, 0]
          : convention === 0
            ? [4, 2, 1, 1]
            : [2, 4, 1, 1]
      );
    }
  } finally {
    fixture.destroy();
  }
});

it('northward segments with either zero sign have zero compass bearing', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'rose-axis-mean');
  try {
    const summary = fixture.output('summary', 'float32', 5);
    const bins = fixture.output('bins', 'float32', 4);
    fixture.graph.add(
      new GPURoseStatistic({
        starts: fixture.input('starts', new Float32Array(4), 'float32x2', 2),
        ends: fixture.input('ends', new Float32Array([0, 1, -0, 1]), 'float32x2', 2),
        circular: {
          mode: 'directional',
          angleUnit: 'degrees',
          origin: 'positive-y-clockwise',
          binCount: 4
        },
        output: {summary: summary.view, bins: bins.view}
      })
    );
    fixture.run();
    const values = await fixture.readFloat32(summary, 5);
    expect(values[1]).toBeCloseTo(0, 5);
    expect(values[2]).toBeCloseTo(1, 5);
    expect(values[4]).toBeCloseTo(2, 5);
  } finally {
    fixture.destroy();
  }
});

it.each([
  72, 360
])('opposite axial compass vectors and cardinal directions share exact %s-bin boundaries', async binCount => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `rose-cardinal-${binCount}`);
  try {
    const starts = fixture.input('starts', new Float32Array(8), 'float32x2', 4);
    const ends = fixture.input(
      'ends',
      new Float32Array([1, 0, -1, 0, 0, 1, 0, -1]),
      'float32x2',
      4
    );
    const results = (['directional', 'axial'] as const).map(mode => {
      const bins = fixture.output(`${mode}-bins`, 'float32', binCount);
      const summary = fixture.output(`${mode}-summary`, 'float32', 5);
      fixture.graph.add(
        new GPURoseStatistic({
          id: mode,
          starts,
          ends,
          circular: {mode, origin: 'positive-y-clockwise', angleUnit: 'radians', binCount},
          output: {bins: bins.view, summary: summary.view}
        })
      );
      return {mode, bins, summary};
    });
    const axisStarts = fixture.input('axis-starts', new Float32Array(4), 'float32x2', 2);
    const axisEnds = fixture.input('axis-ends', new Float32Array([1, 0, -1, 0]), 'float32x2', 2);
    const axisBins = fixture.output('axis-bins', 'float32', binCount);
    const axisSummary = fixture.output('axis-summary', 'float32', 5);
    fixture.graph.add(
      new GPURoseStatistic({
        id: 'axis',
        starts: axisStarts,
        ends: axisEnds,
        circular: {mode: 'axial', origin: 'positive-y-clockwise', angleUnit: 'degrees', binCount},
        output: {bins: axisBins.view, summary: axisSummary.view}
      })
    );
    fixture.run();
    for (const {mode, bins} of results) {
      const expected = new Array<number>(binCount).fill(0);
      if (mode === 'axial') {
        expected[0] = 2;
        expected[binCount / 2] = 2;
      } else
        for (const bin of [0, binCount / 4, binCount / 2, (3 * binCount) / 4]) expected[bin] = 1;
      expect(await fixture.readFloat32(bins, binCount)).toEqual(expected);
    }
    const expected = new Array<number>(binCount).fill(0);
    expected[binCount / 2] = 2;
    expect(await fixture.readFloat32(axisBins, binCount)).toEqual(expected);
    const summary = await fixture.readFloat32(axisSummary, 5);
    expect(summary[1]).toBeCloseTo(90, 5);
    expect(summary[2]).toBeCloseTo(1, 6);
    expect(summary[4]).toBeCloseTo(2, 6);
  } finally {
    fixture.destroy();
  }
});

it('exactly cancelling native-radian directional moments publish a finite zero guarded mean', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'rose-exact-cancellation');
  try {
    const summary = fixture.output('summary', 'float32', 5);
    const bins = fixture.output('bins', 'float32', 4);
    fixture.graph.add(
      new GPURoseStatistic({
        starts: fixture.input('starts', new Float32Array(4), 'float32x2', 2),
        ends: fixture.input('ends', new Float32Array([1, 0, -1, -0]), 'float32x2', 2),
        circular: {
          mode: 'directional',
          angleUnit: 'radians',
          origin: 'positive-x-counter-clockwise',
          binCount: 4
        },
        output: {summary: summary.view, bins: bins.view}
      })
    );
    fixture.run();
    expect(await fixture.readFloat32(summary, 5)).toEqual([2, 0, 0, 1, 0]);
    expect(await fixture.readFloat32(bins, 4)).toEqual([1, 0, 1, 0]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  72, 360
])('non-cardinal circular-angle bins match a CPU oracle with %s bins', async binCount => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `rose-circular-angle-${binCount}`);
  try {
    const directions = [
      [2, 1],
      [-3, 2],
      [-1, -4],
      [2, -5]
    ];
    const weights = [1, 2, 3, 4];
    const starts = fixture.input('starts', new Float32Array(8), 'float32x2', 4);
    const ends = fixture.input('ends', Float32Array.from(directions.flat()), 'float32x2', 4);
    const weightView = fixture.input('weights', Float32Array.from(weights), 'float32', 4);
    const outputs = (['directional', 'axial'] as const).flatMap(mode =>
      (['positive-x-counter-clockwise', 'positive-y-clockwise'] as const).map(origin => {
        const id = `${mode}-${origin}`;
        const bins = fixture.output(`${id}-bins`, 'float32', binCount);
        const summary = fixture.output(`${id}-summary`, 'float32', 5);
        fixture.graph.add(
          new GPURoseStatistic({
            id,
            starts,
            ends,
            weights: weightView,
            circular: {mode, origin, angleUnit: 'radians', binCount},
            output: {bins: bins.view, summary: summary.view}
          })
        );
        const period = mode === 'axial' ? Math.PI : 2 * Math.PI;
        const expected = new Array<number>(binCount).fill(0);
        let cosine = 0;
        let sine = 0;
        for (const [row, [horizontal, vertical]] of directions.entries()) {
          const angle =
            origin === 'positive-y-clockwise'
              ? Math.atan2(horizontal, vertical)
              : Math.atan2(vertical, horizontal);
          const bearing = ((angle % period) + period) % period;
          expected[Math.floor((bearing * binCount) / period)] += weights[row];
          const circularAngle = (mode === 'axial' ? 2 : 1) * bearing;
          cosine += weights[row] * Math.cos(circularAngle);
          sine += weights[row] * Math.sin(circularAngle);
        }
        return {bins, summary, expected, resultant: Math.hypot(cosine, sine) / 10};
      })
    );
    fixture.run();
    for (const {bins, summary, expected, resultant} of outputs) {
      expect(await fixture.readFloat32(bins, binCount)).toEqual(expected);
      const values = await fixture.readFloat32(summary, 5);
      expect(values[0]).toBe(4);
      expect(values[2]).toBeCloseTo(resultant, 5);
    }
  } finally {
    fixture.destroy();
  }
});

it('cooperative group work continues past the first 65,535 workgroups', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'rose-group-stride');
  try {
    const groupCount = 65536;
    const summary = fixture.output('summary', 'float32', groupCount * 5);
    const bins = fixture.output('bins', 'float32', groupCount * 4);
    fixture.graph.add(
      new GPURoseStatistic({
        starts: fixture.input('starts', new Float32Array(8), 'float32x2', 4),
        ends: fixture.input('ends', new Float32Array([1, 0, -1, 0, 0, 1, 0, 1]), 'float32x2', 4),
        groupIds: fixture.input('groups', new Uint32Array([0, 65534, 65535, 65535]), 'uint32', 4),
        groupCount,
        circular: {
          mode: 'directional',
          angleUnit: 'radians',
          origin: 'positive-x-counter-clockwise',
          binCount: 4
        },
        output: {summary: summary.view, bins: bins.view}
      })
    );
    fixture.run();
    const values = await fixture.readFloat32(summary, groupCount * 5);
    expect(values[0]).toBe(1);
    expect(values[5 * 65534]).toBe(1);
    expect(values[5 * 65535]).toBe(2);
    expect(values[5 * 65535 + 1]).toBeCloseTo(Math.PI / 2, 6);
    expect(values[5 * 65535 + 2]).toBeCloseTo(1, 6);
  } finally {
    fixture.destroy();
  }
});
