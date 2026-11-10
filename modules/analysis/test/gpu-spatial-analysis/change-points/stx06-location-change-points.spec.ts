// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  GPULocationChangePoints,
  GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLICE_COUNT,
  GPU_LOCATION_CHANGE_POINTS_NONE,
  GPUPettittTest,
  type GPULocationChangePointCost
} from '../../../src/gpu-spatial-analysis/change-points/index';
import fixture from '../../oracles/fixtures/stx06-change-points.json';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {
  computeChangePointReference,
  computePettittReference,
  createChangePointScene
} from './stx06-change-point-reference';

type LayoutName = 'cell-major' | 'period-major';

const flatten = (rows: Float32Array[], layout: LayoutName): Float32Array => {
  const sliceCount = rows[0].length;
  const cellCount = rows.length;
  const output = new Float32Array(cellCount * sliceCount);
  for (let cell = 0; cell < cellCount; cell++) {
    for (let slice = 0; slice < sliceCount; slice++) {
      output[layout === 'cell-major' ? cell * sliceCount + slice : slice * cellCount + cell] =
        rows[cell][slice];
    }
  }
  return output;
};

async function runChangePoints(options: {
  rows: Float32Array[];
  layout: LayoutName;
  cost: GPULocationChangePointCost;
  penalty: number;
  minimumSegmentLength: number;
  maximumChangePoints: number;
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const {rows, layout, cost, maximumChangePoints} = options;
  const cellCount = rows.length;
  const sliceCount = rows[0].length;
  const changePointCount = rig.output('uint32', cellCount);
  const changePoints = rig.output('uint32', cellCount * maximumChangePoints);
  const requiredChangePointCounts = rig.output('uint32', cellCount);
  const optimalCost = rig.output('float32', cellCount);
  const overflow = rig.output('uint32', 1);
  let contributor: GPULocationChangePoints;
  const parameterValues = new Float32Array(2);
  contributor = new GPULocationChangePoints({
    values: rig.input(flatten(rows, layout), 'float32'),
    layout,
    cellCount,
    sliceCount,
    cost,
    maximumChangePoints,
    parameters: rig.input(parameterValues, 'float32'),
    output: {
      changePointCount,
      changePoints,
      requiredChangePointCounts,
      optimalCost,
      status: {overflow}
    }
  });
  const packed = contributor.getParameterValues({
    penalty: options.penalty,
    minimumSegmentLength: options.minimumSegmentLength
  });
  const parameters = rig.input(packed, 'float32');
  contributor = new GPULocationChangePoints({...contributor.props, parameters});
  rig.run(contributor);
  const result = {
    count: await rig.readUint(changePointCount),
    required: await rig.readUint(requiredChangePointCounts),
    points: await rig.readUint(changePoints),
    cost: await rig.readFloat(optimalCost),
    overflow: (await rig.readUint(overflow))[0]
  };
  rig.destroy();
  return result;
}

const fixtureLocations = (cost: GPULocationChangePointCost) =>
  fixture.locations.map(location => {
    const series = location.series as unknown as Record<string, number[]>;
    return Float32Array.from(series[cost]);
  });

function getPopulationVariance(values: Float32Array): number {
  const finite = Array.from(values).filter(Number.isFinite);
  const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
  return finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length;
}

describe('GPULocationChangePoints against ruptures Pelt', () => {
  for (const cost of ['mean', 'slope', 'mean-variance'] as const) {
    it(`${cost} equals ruptures Pelt on ${fixture.length}-slice fixture series in both layouts`, async () => {
      const expected = fixture.locations.map(location => {
        const entry = location.changePoints as unknown as Record<
          string,
          {
            penalty: number;
            minimumSegmentLength: number;
            breakpoints: number[];
            optimalCost: number;
          }
        >;
        return entry[cost];
      });
      for (const layout of ['cell-major', 'period-major'] as const) {
        const result = await runChangePoints({
          rows: fixtureLocations(cost),
          layout,
          cost,
          penalty: expected[0].penalty,
          minimumSegmentLength: expected[0].minimumSegmentLength,
          maximumChangePoints: 16
        });
        if (!result.count) return;
        expected.forEach((entry, cell) => {
          const count = result.count[cell];
          expect(result.points.slice(cell * 16, cell * 16 + count)).toEqual(entry.breakpoints);
          const tolerance = Math.max(2e-3, Math.abs(entry.optimalCost) * 2e-3);
          if (cost !== 'mean-variance') {
            expect(Math.abs(result.cost[cell] - entry.optimalCost)).toBeLessThan(tolerance);
          }
        });
      }
    });
  }
});

describe('GPULocationChangePoints against the binary64 reference', () => {
  for (const sliceCount of [64, 160]) {
    it(`matches the unpruned dynamic program over ${sliceCount} slices`, async () => {
      const scene = createChangePointScene(sliceCount);
      for (const scenario of scene) {
        const result = await runChangePoints({
          rows: [scenario.values],
          layout: 'cell-major',
          cost: scenario.cost,
          penalty: scenario.penalty,
          minimumSegmentLength: scenario.minimumSegmentLength,
          maximumChangePoints: 32
        });
        if (!result.count) return;
        const expected = computeChangePointReference(
          scenario.cost,
          scenario.values,
          scenario.penalty,
          scenario.minimumSegmentLength
        );
        expect(result.points.slice(0, result.count[0]), scenario.name).toEqual(
          expected.changePoints
        );
        expect(result.required[0]).toBe(expected.changePoints.length);
        expect(result.overflow).toBe(0);
        // The kernel cancels f32 standardized moments of order `sliceCount` in the slope cost, so its
        // absolute error is about 2e-6 * sliceCount in standardized units; scale restores data units.
        const seriesVariance = getPopulationVariance(scenario.values);
        const tolerance =
          scenario.cost === 'slope'
            ? Math.max(2e-3 * Math.abs(expected.optimalCost), 4e-6 * sliceCount * seriesVariance)
            : 2e-3 * Math.max(1, Math.abs(expected.optimalCost));
        expect(Math.abs(result.cost[0] - expected.optimalCost), scenario.name).toBeLessThan(
          tolerance
        );
      }
    });
  }
});

describe('GPULocationChangePoints behavior', () => {
  it('keeps the earliest change points and reports overflow past the capacity', async () => {
    const values = new Float32Array(40);
    for (let slice = 0; slice < 40; slice++) values[slice] = slice % 8 < 4 ? 0 : 50;
    const expected = computeChangePointReference('mean', values, 3, 2);
    const result = await runChangePoints({
      rows: [values],
      layout: 'cell-major',
      cost: 'mean',
      penalty: 3,
      minimumSegmentLength: 2,
      maximumChangePoints: 3
    });
    if (!result.count) return;
    expect(result.count[0]).toBe(3);
    expect(result.required[0]).toBe(expected.changePoints.length);
    expect(expected.changePoints.length).toBeGreaterThan(3);
    expect(result.points).toEqual(expected.changePoints.slice(0, 3));
    expect(result.overflow).toBe(1);
  });

  it('reports no change for constant series, and unused rows hold the sentinel', async () => {
    const constant = new Float32Array(32).fill(4.5);
    const zeroCounts = new Float32Array(32);
    const result = await runChangePoints({
      rows: [constant, zeroCounts],
      layout: 'cell-major',
      cost: 'mean',
      penalty: 1,
      minimumSegmentLength: 2,
      maximumChangePoints: 2
    });
    if (!result.count) return;
    expect(result.count).toEqual([0, 0]);
    expect(result.points).toEqual(new Array(4).fill(GPU_LOCATION_CHANGE_POINTS_NONE));
    expect(result.required).toEqual([0, 0]);
  });

  it('rejects too many slices and a too short minimum segment by name', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const create = (options: {sliceCount?: number; cost?: GPULocationChangePointCost}) => {
      const sliceCount = options.sliceCount ?? 16;
      return new GPULocationChangePoints({
        values: rig.output('float32', sliceCount),
        layout: 'cell-major',
        cellCount: 1,
        sliceCount,
        cost: options.cost ?? 'mean',
        maximumChangePoints: 1,
        parameters: rig.output('float32', 2),
        output: {changePointCount: rig.output('uint32', 1), changePoints: rig.output('uint32', 1)}
      });
    };
    expect(() => create({sliceCount: GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLICE_COUNT + 1})).toThrow(
      /sliceCount/
    );
    expect(() =>
      create({cost: 'variance'}).getParameterValues({penalty: 1, minimumSegmentLength: 1})
    ).toThrow(/minimumSegmentLength/);
    rig.destroy();
  });
});

describe('GPUPettittTest', () => {
  async function runPettitt(rows: Float32Array[]) {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const cellCount = rows.length;
    const statistic = rig.output('uint32', cellCount);
    const changeSlice = rig.output('uint32', cellCount);
    const pValues = rig.output('float32', cellCount);
    rig.run(
      new GPUPettittTest({
        values: rig.input(flatten(rows, 'cell-major'), 'float32'),
        layout: 'cell-major',
        cellCount,
        sliceCount: rows[0].length,
        output: {statistic, changeSlice, pValues}
      })
    );
    const result = {
      statistic: await rig.readUint(statistic),
      changeSlice: await rig.readUint(changeSlice),
      pValues: await rig.readFloat(pValues)
    };
    rig.destroy();
    return result;
  }

  it('matches pyhomogeneity on the fixture series', async () => {
    const rows = fixtureLocations('mean');
    const result = await runPettitt(rows);
    if (!result.statistic) return;
    fixture.pettitt.forEach((entry, cell) => {
      expect(result.statistic[cell]).toBe(entry.u);
      expect(result.changeSlice[cell]).toBe(entry.changeSlice);
      expect(Math.abs(result.pValues[cell] - entry.p)).toBeLessThan(1e-4 * Math.max(entry.p, 1e-3));
    });
  });

  it('matches the binary64 reference with ties and gaps', async () => {
    const tied = Float32Array.from({length: 30}, (_, slice) =>
      slice < 12 ? 1 : slice % 2 === 0 ? 3 : 2
    );
    const gappy = Float32Array.from(tied, (value, slice) => (slice % 5 === 0 ? Number.NaN : value));
    const single = new Float32Array(30).fill(Number.NaN);
    single[9] = 1;
    const result = await runPettitt([tied, gappy, single]);
    if (!result.statistic) return;
    [tied, gappy, single].forEach((row, cell) => {
      const expected = computePettittReference(row);
      expect(result.statistic[cell]).toBe(expected.statistic);
      expect(result.changeSlice[cell]).toBe(expected.changeSlice);
      expect(Math.abs(result.pValues[cell] - expected.pValue)).toBeLessThan(
        1e-4 * Math.max(expected.pValue, 1e-3)
      );
    });
  });
});
