// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  getGPUFiniteTimeLyapunovExponentParameterValues,
  getGPULagrangianFlowMapParameterValues,
  GPUFiniteTimeLyapunovExponent,
  GPULagrangianFlowMap,
  GPU_LAGRANGIAN_NO_REGION
} from '../../../src/raster-analysis/lagrangian-diagnostics/index';
import {GPUTransitionMatrix} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-matrix';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphHarness} from '../../gpu-spatial-analysis/trajectory-staypoints/movement-spec-utils';

type Extent = readonly [number, number, number, number];
type Vector = readonly [number, number];

/** Samples a function at the cell centres of a raster, packed `(u, v)` row-major. */
function sampleFieldFunction(
  width: number,
  height: number,
  extent: Extent,
  velocity: (x: number, y: number) => Vector
): Float32Array {
  const values = new Float32Array(width * height * 2);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const [u, v] = velocity(
        extent[0] + (column + 0.5) * extent[2],
        extent[1] + (row + 0.5) * extent[3]
      );
      values.set([u, v], 2 * (row * width + column));
    }
  }
  return values;
}

/** Binary64 bilinear sampler of the cell-centre convention, `undefined` where invalid. */
function sampleReference(
  field: Float32Array,
  width: number,
  height: number,
  extent: Extent,
  x: number,
  y: number
): Vector | undefined {
  const localX = (x - extent[0]) / extent[2];
  const localY = (y - extent[1]) / extent[3];
  if (!(localX >= 0 && localY >= 0 && localX < width && localY < height)) {
    return undefined;
  }
  const gridX = Math.min(Math.max(localX - 0.5, 0), width - 1);
  const gridY = Math.min(Math.max(localY - 0.5, 0), height - 1);
  const column0 = Math.min(Math.floor(gridX), width - 1);
  const row0 = Math.min(Math.floor(gridY), height - 1);
  const column1 = Math.min(column0 + 1, width - 1);
  const row1 = Math.min(row0 + 1, height - 1);
  const fractionX = gridX - column0;
  const fractionY = gridY - row0;
  const read = (column: number, row: number): Vector => [
    field[2 * (row * width + column)],
    field[2 * (row * width + column) + 1]
  ];
  const corners = [
    read(column0, row0),
    read(column1, row0),
    read(column0, row1),
    read(column1, row1)
  ];
  if (corners.some(corner => Number.isNaN(corner[0]) || Number.isNaN(corner[1]))) {
    return undefined;
  }
  const mix = (axis: 0 | 1): number => {
    const bottom = corners[0][axis] * (1 - fractionX) + corners[1][axis] * fractionX;
    const top = corners[2][axis] * (1 - fractionX) + corners[3][axis] * fractionX;
    return bottom * (1 - fractionY) + top * fractionY;
  };
  return [mix(0), mix(1)];
}

/** Binary64 reference of the flow map: midpoint steps, linear time interpolation of two slices. */
function integrateReference(options: {
  start: Float32Array;
  end: Float32Array;
  width: number;
  height: number;
  extent: Extent;
  seeds: readonly Vector[];
  startTime: number;
  timeStep: number;
  sliceTimes: Vector;
  stepCount: number;
}): (Vector | undefined)[] {
  const {start, end, width, height, extent, timeStep, sliceTimes, stepCount} = options;
  const sample = (x: number, y: number, time: number): Vector | undefined => {
    const weight = Math.min(
      Math.max((time - sliceTimes[0]) / (sliceTimes[1] - sliceTimes[0]), 0),
      1
    );
    const first = sampleReference(start, width, height, extent, x, y);
    const second = sampleReference(end, width, height, extent, x, y);
    if (!first || !second) {
      return undefined;
    }
    return [
      first[0] * (1 - weight) + second[0] * weight,
      first[1] * (1 - weight) + second[1] * weight
    ];
  };
  return options.seeds.map(seed => {
    let [x, y] = seed;
    for (let step = 0; step < stepCount; step++) {
      const time = options.startTime + step * timeStep;
      const first = sample(x, y, time);
      if (!first) {
        return undefined;
      }
      const second = sample(
        x + 0.5 * timeStep * first[0],
        y + 0.5 * timeStep * first[1],
        time + 0.5 * timeStep
      );
      if (!second) {
        return undefined;
      }
      x += timeStep * second[0];
      y += timeStep * second[1];
    }
    return sampleReference(start, width, height, extent, x, y) ? [x, y] : undefined;
  });
}

/** Runs the flow map, and the FTLE when a seed grid is given, and reads the results. */
async function runFlowMap(options: {
  start: Float32Array;
  end: Float32Array;
  width: number;
  height: number;
  extent: Extent;
  seeds: Float32Array;
  startTime: number;
  timeStep: number;
  sliceTimes: Vector;
  stepCount: number;
  seedGrid?: {width: number; height: number; spacingX: number; spacingY: number};
  regions?: Uint32Array;
  connectivityClassCount?: number;
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'mob06-test');
  const {width, height, stepCount} = options;
  const seedCount = options.seeds.length / 2;
  const seeds = harness.input('seeds', options.seeds, 'float32x2', seedCount);
  const start = harness.input('start', options.start, 'float32x2', width * height);
  const end = harness.input('end', options.end, 'float32x2', width * height);
  const parameters = harness.input(
    'parameters',
    getGPULagrangianFlowMapParameterValues({
      fieldExtent: options.extent,
      startTime: options.startTime,
      timeStep: options.timeStep,
      sliceStartTime: options.sliceTimes[0],
      sliceEndTime: options.sliceTimes[1]
    }),
    'float32',
    8
  );
  const positions = harness.output('positions', 'float32x2', seedCount);
  const regions = options.regions
    ? harness.input('regions', options.regions, 'uint32', width * height)
    : undefined;
  const regionClasses = options.regions
    ? harness.output('region-classes', 'uint32', 2 * seedCount)
    : undefined;
  const residence = options.regions ? harness.output('residence', 'float32', seedCount) : undefined;
  harness.graph.add(
    new GPULagrangianFlowMap({
      seeds,
      velocitiesStart: start,
      velocitiesEnd: end,
      fieldWidth: width,
      fieldHeight: height,
      stepCount,
      parameters,
      regions,
      output: {
        positions: positions.view,
        regionClasses: regionClasses?.view,
        residenceTimes: residence?.view
      }
    })
  );
  let ftle: ReturnType<typeof harness.output<'float32'>> | undefined;
  if (options.seedGrid) {
    const grid = options.seedGrid;
    const ftleParameters = harness.input(
      'ftle-parameters',
      getGPUFiniteTimeLyapunovExponentParameterValues({
        seedSpacingX: grid.spacingX,
        seedSpacingY: grid.spacingY,
        integrationTime: stepCount * options.timeStep
      }),
      'float32',
      4
    );
    ftle = harness.output('ftle', 'float32', seedCount);
    harness.graph.add(
      new GPUFiniteTimeLyapunovExponent({
        positions: positions.view,
        seedWidth: grid.width,
        seedHeight: grid.height,
        parameters: ftleParameters,
        output: {ftle: ftle.view}
      })
    );
  }
  let connectivity: ReturnType<typeof harness.output<'uint32'>> | undefined;
  if (regionClasses && options.connectivityClassCount) {
    connectivity = harness.output(
      'connectivity',
      'uint32',
      options.connectivityClassCount * options.connectivityClassCount
    );
    harness.graph.add(
      new GPUTransitionMatrix({
        classes: regionClasses.view,
        periodCount: 2,
        classCount: options.connectivityClassCount,
        output: {counts: connectivity.view}
      })
    );
  }
  return harness.run(async () => ({
    positions: await readFloat32(positions.buffer, 2 * seedCount),
    ftle: ftle ? await readFloat32(ftle.buffer, seedCount) : undefined,
    regionClasses: regionClasses
      ? await readUint32(regionClasses.buffer, 2 * seedCount)
      : undefined,
    residence: residence ? await readFloat32(residence.buffer, seedCount) : undefined,
    connectivity: connectivity
      ? await readUint32(connectivity.buffer, options.connectivityClassCount! ** 2)
      : undefined
  }));
}

/** A regular `columns` by `rows` seed grid over `[-half, half]`, row-major with row 0 at smallest y. */
function createSeedGrid(columns: number, rows: number, half: number) {
  const spacing = (2 * half) / (columns - 1);
  const seeds = new Float32Array(columns * rows * 2);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      seeds.set([-half + column * spacing, -half + row * spacing], 2 * (row * columns + column));
    }
  }
  return {seeds, spacing};
}

const STRAIN_EXTENT: Extent = [-8, -8, 0.5, 0.5];
const STRAIN_SIZE = 32;

describe('GPULagrangianFlowMap and GPUFiniteTimeLyapunovExponent (MOB-06)', () => {
  it('recovers the exponent of a steady linear strain field, forward', async () => {
    // u = e x, v = -e y: the flow map is (x exp(eT), y exp(-eT)), so FTLE = e everywhere.
    const strain = 0.8;
    const field = sampleFieldFunction(STRAIN_SIZE, STRAIN_SIZE, STRAIN_EXTENT, (x, y) => [
      strain * x,
      -strain * y
    ]);
    const {seeds, spacing} = createSeedGrid(11, 11, 1);
    const result = await runFlowMap({
      start: field,
      end: field,
      width: STRAIN_SIZE,
      height: STRAIN_SIZE,
      extent: STRAIN_EXTENT,
      seeds,
      startTime: 0,
      timeStep: 0.01,
      sliceTimes: [0, 1],
      stepCount: 100,
      seedGrid: {width: 11, height: 11, spacingX: spacing, spacingY: spacing}
    });
    for (const exponent of result.ftle!) {
      expect(exponent).toBeCloseTo(strain, 3);
    }
    // The flow map itself: the seed (1, 1) ends at (exp(0.8), exp(-0.8)) up to the RK2 error.
    const last = seeds.length / 2 - 1;
    expect(result.positions[2 * last]).toBeCloseTo(Math.exp(strain), 3);
    expect(result.positions[2 * last + 1]).toBeCloseTo(Math.exp(-strain), 3);
  });

  it('separates forward from backward exponents of a one-directional stretch', async () => {
    // u = e x, v = 0: forward the x axis stretches by exp(eT) (FTLE = e); backward it contracts and
    // the larger singular value of the backward map is 1 (FTLE = 0).
    const stretch = 0.5;
    const field = sampleFieldFunction(STRAIN_SIZE, STRAIN_SIZE, STRAIN_EXTENT, x => [
      stretch * x,
      0
    ]);
    const {seeds, spacing} = createSeedGrid(11, 11, 1);
    const common = {
      start: field,
      end: field,
      width: STRAIN_SIZE,
      height: STRAIN_SIZE,
      extent: STRAIN_EXTENT,
      seeds,
      startTime: 0,
      sliceTimes: [0, 1] as Vector,
      stepCount: 100,
      seedGrid: {width: 11, height: 11, spacingX: spacing, spacingY: spacing}
    };
    const forward = await runFlowMap({...common, timeStep: 0.01});
    const backward = await runFlowMap({...common, timeStep: -0.01});
    for (const exponent of forward.ftle!) {
      expect(exponent).toBeCloseTo(stretch, 3);
    }
    for (const exponent of backward.ftle!) {
      expect(Math.abs(exponent)).toBeLessThan(1e-3);
    }
  });

  it('interpolates two slices in time: the exponent is the time mean of a ramped strain rate', async () => {
    // a(t) ramps linearly from 0 to 2 across the slices, u = a(t) x, v = -a(t) y. The flow map has
    // stretch exp(int a dt) = exp(1) over T = 1, so FTLE = 1. A frozen first slice would give 0.
    const slice = (rate: number) =>
      sampleFieldFunction(STRAIN_SIZE, STRAIN_SIZE, STRAIN_EXTENT, (x, y) => [rate * x, -rate * y]);
    const {seeds, spacing} = createSeedGrid(11, 11, 1);
    const common = {
      width: STRAIN_SIZE,
      height: STRAIN_SIZE,
      extent: STRAIN_EXTENT,
      seeds,
      startTime: 0,
      timeStep: 0.01,
      sliceTimes: [0, 1] as Vector,
      stepCount: 100,
      seedGrid: {width: 11, height: 11, spacingX: spacing, spacingY: spacing}
    };
    const ramped = await runFlowMap({...common, start: slice(0), end: slice(2)});
    for (const exponent of ramped.ftle!) {
      expect(exponent).toBeCloseTo(1, 3);
    }
    const frozen = await runFlowMap({...common, start: slice(0), end: slice(0)});
    for (const exponent of frozen.ftle!) {
      expect(Math.abs(exponent)).toBeLessThan(1e-3);
    }
    // Shifting the release time into the second half of the slice interval: a runs 1 to 2 over
    // T = 0.5, mean 1.5.
    const late = await runFlowMap({
      ...common,
      start: slice(0),
      end: slice(2),
      startTime: 0.5,
      stepCount: 50
    });
    for (const exponent of late.ftle!) {
      expect(exponent).toBeCloseTo(1.5, 3);
    }
  });

  it('matches a binary64 reference on a time-varying field with a missing cell and lost particles', async () => {
    const width = 24;
    const height = 20;
    const extent: Extent = [0, 0, 0.4, 0.4];
    const start = sampleFieldFunction(width, height, extent, (x, y) => [
      -0.3 * (y - 4) + 0.2 * Math.sin(x),
      0.3 * (x - 4.8) + 0.1 * Math.cos(y)
    ]);
    const end = sampleFieldFunction(width, height, extent, (x, y) => [
      0.5 * (y - 4) - 0.1 * Math.sin(2 * x),
      -0.4 * (x - 4.8) + 0.3
    ]);
    // A missing cell in a corner region.
    start.set([NaN, NaN], 2 * (3 * width + 20));
    end.set([NaN, NaN], 2 * (3 * width + 20));
    const seedList: Vector[] = [];
    for (let index = 0; index < 60; index++) {
      seedList.push([0.2 + ((index * 37) % 97) * 0.045, 0.2 + ((index * 53) % 89) * 0.04]);
    }
    seedList.push([-1, 2]);
    const seeds = Float32Array.from(seedList.flat());
    const options = {
      start,
      end,
      width,
      height,
      extent,
      startTime: 0.25,
      timeStep: 0.04,
      sliceTimes: [0, 3] as Vector,
      stepCount: 60
    };
    const expected = integrateReference({...options, seeds: seedList});
    const actual = await runFlowMap({...options, seeds});
    let lostCount = 0;
    let keptCount = 0;
    expected.forEach((position, index) => {
      const x = actual.positions[2 * index];
      const y = actual.positions[2 * index + 1];
      if (!position) {
        lostCount++;
        expect(Number.isNaN(x) && Number.isNaN(y)).toBe(true);
      } else {
        keptCount++;
        expect(Math.abs(x - position[0])).toBeLessThan(2e-3);
        expect(Math.abs(y - position[1])).toBeLessThan(2e-3);
      }
    });
    // The case must exercise both outcomes.
    expect(lostCount).toBeGreaterThan(3);
    expect(keptCount).toBeGreaterThan(10);
  });

  it('reports residence times and the connectivity matrix of a uniform flow through three regions', async () => {
    // Uniform u = 1 on a 40 by 4 field of unit cells. Regions by column: 0 below 8, 1 below 24, 2 above.
    const width = 40;
    const height = 4;
    const extent: Extent = [0, 0, 1, 1];
    const field = sampleFieldFunction(width, height, extent, () => [1, 0]);
    const regions = new Uint32Array(width * height);
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        regions[row * width + column] = column < 8 ? 0 : column < 24 ? 1 : 2;
      }
    }
    const originsX = [
      ...Array.from({length: 8}, (_, index) => index + 0.52),
      ...Array.from({length: 8}, (_, index) => 16 + index + 0.52)
    ];
    const seeds = Float32Array.from(originsX.flatMap(x => [x, 2]));
    const timeStep = 0.05;
    const result = await runFlowMap({
      start: field,
      end: field,
      width,
      height,
      extent,
      seeds,
      startTime: 0,
      timeStep,
      sliceTimes: [0, 1],
      stepCount: 200,
      regions,
      connectivityClassCount: 3
    });
    const seedCount = originsX.length;
    originsX.forEach((x, index) => {
      const startRegion = x < 8 ? 0 : 1;
      const boundary = startRegion === 0 ? 8 : 24;
      expect(result.regionClasses![index]).toBe(startRegion);
      expect(result.regionClasses![seedCount + index]).toBe(startRegion === 0 ? 1 : 2);
      // Steps that begin inside the region: positions x + k h < boundary.
      expect(result.residence![index]).toBeCloseTo(
        Math.ceil((boundary - x) / timeStep) * timeStep,
        4
      );
    });
    // Counts[from * 3 + to]: eight 0 -> 1 and eight 1 -> 2 transitions.
    expect(result.connectivity).toEqual([0, 8, 0, 0, 0, 8, 0, 0, 0]);
  });

  it('marks a particle that leaves the field as lost: no end region, residence up to the exit', async () => {
    const width = 8;
    const height = 4;
    const extent: Extent = [0, 0, 1, 1];
    const field = sampleFieldFunction(width, height, extent, () => [1, 0]);
    const regions = new Uint32Array(width * height).fill(0);
    const seeds = Float32Array.from([6.52, 2, 100, 100]);
    const result = await runFlowMap({
      start: field,
      end: field,
      width,
      height,
      extent,
      seeds,
      startTime: 0,
      timeStep: 0.1,
      sliceTimes: [0, 1],
      stepCount: 50,
      regions
    });
    expect(result.positions.slice(0, 2).every(Number.isNaN)).toBe(true);
    expect(result.regionClasses).toEqual([
      0,
      GPU_LAGRANGIAN_NO_REGION,
      GPU_LAGRANGIAN_NO_REGION,
      GPU_LAGRANGIAN_NO_REGION
    ]);
    expect(result.residence![0]).toBeCloseTo(Math.ceil((8 - 6.52) / 0.1) * 0.1, 4);
    // A seed outside the field has no region and no residence time.
    expect(result.residence![1]).toBe(0);
  });
});
