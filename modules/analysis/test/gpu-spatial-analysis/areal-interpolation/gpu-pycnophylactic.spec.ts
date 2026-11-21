// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getGPUPolygonRasterizationParameterValues} from '../../../src/raster-analysis/polygon-rasterization/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {GPUPycnophylacticInterpolation} from '../../../src/gpu-spatial-analysis/areal-interpolation/index';
import {computePycnophylacticOracle, createBlockZones} from './areal-interpolation-oracle';
import {expectClose, GraphRig} from './areal-interpolation-harness';
import {PYCNOPHYLACTIC_NUMPY} from './change-of-support-numpy-fixtures';
import {requireTestDevice} from '../../../src/testing/require-test-device';

const WIDTH = 20;
const HEIGHT = 15;

async function runPycnophylactic(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  iterations: number,
  kernel: 'rook' | 'box'
) {
  const {zones, zoneCount} = createBlockZones(WIDTH, HEIGHT, 5, 5, {holeSeed: 3, holeRate: 0.06});
  // A skewed field of totals: a peak next to empty zones forces the non-negativity clamp.
  const totals = Float32Array.from({length: zoneCount}, (_, index) =>
    index === 5 ? 900 : index % 3 === 0 ? 0 : 40 + index * 10
  );
  const rig = new GraphRig(device);
  const output = rig.output('float32', WIDTH * HEIGHT);
  rig.run(
    new GPUPycnophylacticInterpolation({
      grid: makeGPURasterMetadata({width: WIDTH, height: HEIGHT}),
      zones: rig.input(zones, 'uint32'),
      zoneCount,
      totals: rig.input(totals, 'float32'),
      maximumIterations: iterations,
      kernel,
      output: {mass: output.view}
    })
  );
  const actual = await output.readFloat32();
  rig.destroy();
  const expected = computePycnophylacticOracle({
    width: WIDTH,
    height: HEIGHT,
    zones,
    zoneCount,
    totals,
    iterations,
    kernel
  });
  return {actual, expected, zones, zoneCount, totals};
}

for (const kernel of ['rook', 'box'] as const) {
  for (const iterations of [0, 1, 6, 7]) {
    it(`GPUPycnophylacticInterpolation matches the oracle (${kernel}, ${iterations} iterations)`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {actual, expected, zones, zoneCount, totals} = await runPycnophylactic(
        device,
        iterations,
        kernel
      );
      expectClose(actual, expected, 'pycnophylactic surface', 2e-3, 1e-3);
      expect(actual.some(value => value > 0)).toBe(true);
      // Mass preservation per zone and non-negativity.
      const sums = new Array<number>(zoneCount).fill(0);
      for (let cell = 0; cell < actual.length; cell++) {
        expect(actual[cell]).toBeGreaterThanOrEqual(0);
        if (zones[cell] < zoneCount) sums[zones[cell]] += actual[cell];
        else expect(actual[cell]).toBe(0);
      }
      const present = new Set<number>(Array.from(zones).filter(zone => zone < zoneCount));
      for (const zone of present) {
        expect(sums[zone]).toBeCloseTo(totals[zone], 2);
      }
    });
  }
}

it('GPUPycnophylacticInterpolation (box) smooths the surface across zone borders', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const flat = await runPycnophylactic(device, 0, 'rook');
  const smooth = await runPycnophylactic(device, 6, 'box');
  const roughness = (values: number[]) => {
    let total = 0;
    for (let row = 0; row < HEIGHT; row++) {
      for (let column = 0; column + 1 < WIDTH; column++) {
        total += (values[row * WIDTH + column] - values[row * WIDTH + column + 1]) ** 2;
      }
    }
    return total;
  };
  expect(roughness(smooth.actual)).toBeLessThan(roughness(flat.actual));
});

// Oracle: <scratchpad>/fix/work/D/change-of-support/pycnophylactic.py (numpy 2.5.4, float64),
// pinned in change-of-support-numpy-fixtures.ts and reproduced by the TS oracle in the node spec.
/** Reverses the row order of a row-major raster. */
function mirrorRows(values: ArrayLike<number>, width: number, height: number): number[] {
  const mirrored: number[] = [];
  for (let row = height - 1; row >= 0; row--) {
    for (let column = 0; column < width; column++) mirrored.push(values[row * width + column]);
  }
  return mirrored;
}

async function runNumpyFixture(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  options: {
    iterations: number;
    anchors: boolean;
    tolerance?: number;
    geographic?: boolean;
    units?: 'meters' | 'kilometers';
    density?: boolean;
    /** Store rows north-up (row 0 northernmost): the same raster with its rows mirrored. */
    northUp?: boolean;
  }
) {
  const {fixture} = PYCNOPHYLACTIC_NUMPY;
  const cellCount = fixture.width * fixture.height;
  const rig = new GraphRig(device);
  const output = rig.output('float32', cellCount);
  const density = options.density ? rig.output('float32', cellCount) : undefined;
  const converged = rig.output('uint32', 1);
  const iterationCount = rig.output('uint32', 1);
  const invalid = rig.output('uint32', 1);
  const geographic = PYCNOPHYLACTIC_NUMPY.geographic;
  // The fixture rows advance north from the origin; a north-up grid starts at the top edge and
  // stores the mirrored rows.
  const grid = makeGPURasterMetadata({
    width: fixture.width,
    height: fixture.height,
    origin: [
      10,
      geographic.originLatitude + (options.northUp ? fixture.height * geographic.cellHeight : 0)
    ],
    cellSize: [geographic.cellWidth, geographic.cellHeight],
    rowDirection: options.northUp ? 'south' : 'north'
  });
  const order = (values: ArrayLike<number>) =>
    options.northUp ? mirrorRows(values, fixture.width, fixture.height) : Array.from(values);
  rig.run(
    new GPUPycnophylacticInterpolation({
      grid,
      zones: rig.input(Uint32Array.from(order(fixture.zones)), 'uint32'),
      zoneCount: fixture.zoneCount,
      totals: rig.input(Float32Array.from(fixture.totals), 'float32'),
      maximumIterations: options.iterations,
      output: {
        mass: output.view,
        density: density?.view,
        status: {
          converged: options.tolerance === undefined ? undefined : converged.view,
          iterationCount: options.tolerance === undefined ? undefined : iterationCount.view,
          invalidCount: invalid.view
        }
      },
      zoneAnchorCells: options.anchors
        ? rig.input(
            Uint32Array.from(
              fixture.anchors.map(cell =>
                options.northUp && cell < fixture.width * fixture.height
                  ? (fixture.height - 1 - Math.floor(cell / fixture.width)) * fixture.width +
                    (cell % fixture.width)
                  : cell
              )
            ),
            'uint32'
          )
        : undefined,
      tolerance:
        options.tolerance === undefined
          ? undefined
          : rig.input(Float32Array.of(options.tolerance), 'float32'),
      ...(options.geographic
        ? {
            spatialContext: {
              coordinateSpace: 'longitude-latitude',
              metric: 'great-circle',
              units: options.units
            } as const,
            // Packed placement: [10, originLatitude, cellWidth, +-cellHeight] as before.
            parameters: rig.input(getGPUPolygonRasterizationParameterValues(grid), 'float32')
          }
        : {})
    })
  );
  const restore = (values: ArrayLike<number>) =>
    options.northUp ? mirrorRows(values, fixture.width, fixture.height) : Array.from(values);
  const result = {
    mass: restore(await output.readFloat32()),
    density: density ? restore(await density.readFloat32()) : undefined,
    converged: (await converged.readUint32())[0],
    iterationCount: (await iterationCount.readUint32())[0],
    invalid: (await invalid.readUint32())[0]
  };
  rig.destroy();
  return result;
}

it('GPUPycnophylacticInterpolation anchors keep the mass of zones without cells (numpy)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const without = await runNumpyFixture(device, {iterations: 6, anchors: false});
  expectClose(without.mass, PYCNOPHYLACTIC_NUMPY.planar6.mass, 'unanchored', 2e-3, 1e-3);
  expect(without.invalid).toBe(PYCNOPHYLACTIC_NUMPY.planar6.invalidCount);
  const anchored = await runNumpyFixture(device, {iterations: 6, anchors: true});
  expectClose(anchored.mass, PYCNOPHYLACTIC_NUMPY.planarAnchored6.mass, 'anchored', 2e-3, 1e-3);
  expect(anchored.invalid).toBe(0);
  expect(anchored.mass.reduce((sum, value) => sum + value, 0)).toBeCloseTo(507, 1);
});

it('GPUPycnophylacticInterpolation stops at the tolerance and reports status (numpy)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const stopped = await runNumpyFixture(device, {
    // Stops at six, leaving seven unused iterations: the final surface is in scratch.
    iterations: 13,
    anchors: true,
    tolerance: PYCNOPHYLACTIC_NUMPY.tolerance
  });
  expectClose(stopped.mass, PYCNOPHYLACTIC_NUMPY.planarTolerance.mass, 'tolerance', 2e-3, 1e-3);
  expect([stopped.converged, stopped.iterationCount]).toEqual([
    1,
    PYCNOPHYLACTIC_NUMPY.planarTolerance.flaggedAt
  ]);
  const totals = new Array<number>(6).fill(0);
  for (const [cell, zone] of PYCNOPHYLACTIC_NUMPY.planarTolerance.zones.entries()) {
    if (zone < totals.length) totals[zone] += stopped.mass[cell];
  }
  expectClose(totals, [120, 0, 45, 300, 30, 12], 'stopped zone totals', 1e-5, 1e-4);
  const limited = await runNumpyFixture(device, {iterations: 3, anchors: true, tolerance: 1e-12});
  expectClose(limited.mass, PYCNOPHYLACTIC_NUMPY.planarLimit.mass, 'limit', 2e-3, 1e-3);
  expect([limited.converged, limited.iterationCount]).toEqual([0, 3]);
});

it('GPUPycnophylacticInterpolation publishes a hand-worked stopped surface with an odd unused budget (GEO-130)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const mass = rig.output('float32', 4);
  const converged = rig.output('uint32', 1);
  const iterationCount = rig.output('uint32', 1);
  try {
    rig.run(
      new GPUPycnophylacticInterpolation({
        grid: makeGPURasterMetadata({width: 4, height: 1}),
        zones: rig.input(Uint32Array.of(0, 0, 1, 1), 'uint32'),
        zoneCount: 2,
        totals: rig.input(Float32Array.of(2, 6), 'float32'),
        kernel: 'box',
        maximumIterations: 5,
        tolerance: rig.input(Float32Array.of(0.1), 'float32'),
        output: {
          mass: mass.view,
          status: {converged: converged.view, iterationCount: iterationCount.view}
        }
      })
    );
    // Hand calculation with box averages (including self) and additive zone restoration:
    // initial [1,1,3,3]; step 1 averages [1,5/3,7/3,3], restores to [2/3,4/3,8/3,10/3].
    // Step 2 averages [1,14/9,22/9,3], restores by [-5/18,-5/18,5/18,5/18].
    // Changes relative to zone means 1 and 3: max step 1 = 1/3 > .1;
    // max step 2 = 1/18 < .1. Stop at two, with three unused iterations.
    const expected = [
      0.7222222222222222, 1.2777777777777777, 2.7222222222222223, 3.2777777777777777
    ];
    for (let submission = 0; submission < 2; submission++) {
      if (submission) rig.resubmit();
      const actual = await mass.readFloat32();
      for (const value of actual) expect(Number.isFinite(value)).toBe(true);
      expectClose(actual, expected, 'hand-worked stopped surface', 1e-6, 1e-6);
      expectClose(
        [actual[0] + actual[1], actual[2] + actual[3]],
        [2, 6],
        'zone totals',
        1e-6,
        1e-6
      );
      expect(await converged.readUint32()).toEqual([1]);
      expect(await iterationCount.readUint32()).toEqual([2]);
    }
  } finally {
    rig.destroy();
  }
});

it('GPUPycnophylacticInterpolation smooths density per spherical cell area on a longitude/latitude raster (numpy)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runNumpyFixture(device, {
    iterations: 6,
    anchors: true,
    geographic: true,
    density: true
  });
  expectClose(result.mass, PYCNOPHYLACTIC_NUMPY.geographic6.mass, 'geographic mass', 2e-3, 1e-3);
  expectClose(
    result.density!,
    PYCNOPHYLACTIC_NUMPY.geographic6.density,
    'geographic density',
    2e-3,
    1e-12
  );
  expect(result.mass.reduce((sum, value) => sum + value, 0)).toBeCloseTo(507, 1);
});

it('GPUPycnophylacticInterpolation gives a north-up longitude/latitude grid the mirrored result of a north-advancing one', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const options = {iterations: 6, anchors: true, geographic: true, density: true};
  const northAdvancing = await runNumpyFixture(device, options);
  const northUp = await runNumpyFixture(device, {...options, northUp: true});
  // Cell areas follow the mirrored rows, so the result is the same raster stored upside down.
  expectClose(northUp.mass, northAdvancing.mass, 'north-up mass', 1e-4, 1e-4);
  expectClose(northUp.density!, northAdvancing.density!, 'north-up density', 1e-4, 1e-14);
  expect(northUp.mass.reduce((sum, value) => sum + value, 0)).toBeCloseTo(507, 1);
});

it('GPUPycnophylacticInterpolation preserves mass and scales geographic density to square kilometers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const options = {iterations: 6, anchors: true, geographic: true, density: true};
  const meters = await runNumpyFixture(device, {...options, units: 'meters'});
  const kilometers = await runNumpyFixture(device, {...options, units: 'kilometers'});
  expectClose(kilometers.mass, meters.mass, 'kilometers mass', 1e-4, 1e-4);
  expectClose(
    kilometers.density!,
    meters.density!.map(value => value * 1e6),
    'density per square kilometer',
    1e-4,
    1e-8
  );
});
