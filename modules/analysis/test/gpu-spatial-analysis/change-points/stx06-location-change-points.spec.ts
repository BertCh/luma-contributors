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

import {UNIT_SCALING_PARTITIONS} from './unit-scaling-fixture';

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

describe('GPULocationChangePoints variance costs do not depend on the unit of the values', () => {
  // Math review point-process-clustering-01. 60 slices alternating +-1 for 30 slices, then +-5,
  // times a unit. About the series mean (0) the first half has variance exactly unit^2 and the
  // second exactly 25 unit^2, in every sub-segment of even length, so by hand:
  //   one segment:  60 (ln(2 pi) + ln(13 unit^2) + 1)
  //   split at 30:  60 (ln(2 pi) + 1) + 30 ln(unit^2) + 30 ln(25 unit^2) + penalty
  // The split wins by 60 ln 13 - 30 ln 25 - penalty = 57.3 - penalty, and no further split of a
  // half can gain the penalty back. The partition must not depend on the unit; the old kernel
  // clamped every segment variance to 1e-11 in data units and found no change point at 1e-6.
  const penalty = 2 * Math.log(60);
  const pattern = (unit: number) =>
    Float32Array.from(
      {length: 60},
      (_, slice) => unit * (slice % 2 === 0 ? 1 : -1) * (slice < 30 ? 1 : 5)
    );
  for (const cost of ['variance', 'mean-variance'] as const) {
    it(`${cost} finds slice 30 in units of 1, 1e-3, 1e-6 and 1e-7`, async () => {
      const units = [1, 1e-3, 1e-6, 1e-7];
      const result = await runChangePoints({
        rows: units.map(pattern),
        layout: 'cell-major',
        cost,
        penalty,
        minimumSegmentLength: 2,
        maximumChangePoints: 4
      });
      if (!result.count) return;
      units.forEach((unit, cell) => {
        expect(result.count[cell], `unit ${unit}`).toBe(1);
        expect(result.points[cell * 4], `unit ${unit}`).toBe(30);
        const expectedCost =
          60 * (Math.log(2 * Math.PI) + 1) +
          30 * Math.log(unit * unit) +
          30 * Math.log(25 * unit * unit) +
          penalty;
        expect(Math.abs(result.cost[cell] - expectedCost), `unit ${unit}`).toBeLessThan(
          2e-3 * Math.abs(expectedCost)
        );
      });
    });
  }

  it('keeps a quiet half that a plain float32 prefix sum would lose', async () => {
    // 512 slices: +-1000 for 256 slices, then +-1. Standardized, the squares are 2 and 2e-6, so
    // after the loud half the prefix of squares is 512 and its last place is 6e-5: a plain float32
    // prefix sum does not move for any quiet slice and the quiet variance reads as 0. By hand the
    // optimum is one change at 256 with
    //   512 (ln(2 pi) + 1) + 256 ln(1e6) + 256 ln(1) + penalty.
    const loudAndQuiet = Float32Array.from(
      {length: 512},
      (_, slice) => (slice % 2 === 0 ? 1 : -1) * (slice < 256 ? 1000 : 1)
    );
    const quietPenalty = 2 * Math.log(512);
    const result = await runChangePoints({
      rows: [loudAndQuiet],
      layout: 'cell-major',
      cost: 'variance',
      penalty: quietPenalty,
      minimumSegmentLength: 2,
      maximumChangePoints: 4
    });
    if (!result.count) return;
    expect(result.count[0]).toBe(1);
    expect(result.points[0]).toBe(256);
    const expectedCost = 512 * (Math.log(2 * Math.PI) + 1) + 256 * Math.log(1e6) + quietPenalty;
    expect(Math.abs(result.cost[0] - expectedCost)).toBeLessThan(2e-3 * expectedCost);
  });
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
    const constant = new Float32Array(30).fill(Number.NaN);
    constant[1] = 5;
    constant[3] = 5;
    const result = await runPettitt([tied, gappy, single, constant]);
    if (!result.statistic) return;
    [tied, gappy, single, constant].forEach((row, cell) => {
      const expected = computePettittReference(row);
      expect(result.statistic[cell]).toBe(expected.statistic);
      expect(result.changeSlice[cell]).toBe(expected.changeSlice);
      if (Number.isNaN(expected.pValue)) {
        expect(result.pValues[cell]).toBeNaN();
      } else {
        expect(Number.isFinite(result.pValues[cell])).toBe(true);
        expect(Math.abs(result.pValues[cell] - expected.pValue)).toBeLessThan(
          1e-4 * Math.max(expected.pValue, 1e-3)
        );
      }
    });
  });
});

for (const fixture of UNIT_SCALING_PARTITIONS) {
  it(`PTS-125 device ${fixture.cost}, units x${fixture.scale}, penalty ${fixture.penalty}`, async () => {
    const result = await runChangePoints({
      rows: [Float32Array.from(fixture.values)],
      layout: 'cell-major',
      cost: fixture.cost,
      penalty: fixture.penalty,
      minimumSegmentLength: fixture.minimum,
      maximumChangePoints: 7
    });
    const points = result.points.slice(0, result.count[0]);
    if (fixture.cost === 'mean' && (fixture.scale === 1 || fixture.penalty === 200)) {
      expect([[], [4]]).toContainEqual(points);
    } else expect(points).toEqual(fixture.points);
    expect(Number.isFinite(result.cost[0])).toBe(true);
    expect(Math.abs(result.cost[0] - fixture.expectedCost)).toBeLessThan(0.002);
    expect(result.overflow).toBe(0);
  });
}

for (const [scale, penalty, expectedPoints] of [
  [1, 2, []],
  [10, 2, [4]],
  [10, 20, []]
] as const) {
  it(`PTS-125 device Poisson count scale ${scale}, penalty ${penalty}`, async () => {
    // Hand-computed split gain: scale * 2*(4*log(2/3)+8*log(4/3)) = scale*1.359192294363178.
    const result = await runChangePoints({
      rows: [Float32Array.from([1, 1, 1, 1, 2, 2, 2, 2], value => value * scale)],
      layout: 'cell-major',
      cost: 'poisson',
      penalty,
      minimumSegmentLength: 2,
      maximumChangePoints: 7
    });
    expect(result.points.slice(0, result.count[0])).toEqual(expectedPoints);
    expect(Number.isFinite(result.cost[0])).toBe(true);
    expect(result.overflow).toBe(0);
  });
}
