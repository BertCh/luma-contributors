// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPU_POINT_CLOUD_CHANGE_STATUS_GRID_INCOMPLETE,
  GPUPointCloudChange
} from '@luma-contributors/analysis/pointcloud-analysis';
import type {GPUPointCloudChangeProps} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import reference from '../oracles/fixtures/pdal-m3c2-upstream-golden.json';

type Triple = readonly [number, number, number];

function toTriples(rows: number[][]): Triple[] {
  return rows.map(([x, y, z]): Triple => [x, y, z]);
}
type Options = Partial<
  Pick<
    GPUPointCloudChangeProps,
    | 'samplingCovariance'
    | 'registrationError'
    | 'registrationCovariance'
    | 'uncertaintyCombination'
    | 'minimumSupport'
  >
> & {
  alignment?: Float32Array;
  radius: number;
  halfDepth: number;
};

/** Mulberry32, a small deterministic generator, so the random cloud is the same on every run. */
function createRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Grid that contains every point with a margin, with the given cell size. */
function getGrid(points: readonly Triple[], cellSize: number, margin: number) {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (const point of points) {
    for (let axis = 0; axis < 3; axis++) {
      minimum[axis] = Math.min(minimum[axis], point[axis]);
      maximum[axis] = Math.max(maximum[axis], point[axis]);
    }
  }
  return {
    origin: minimum.map(value => value - margin) as unknown as Triple,
    cellSize,
    dimensions: minimum.map(
      (value, axis) => Math.ceil((maximum[axis] - value + 2 * margin) / cellSize) + 1
    ) as unknown as Triple
  };
}

/** Runs the unindexed and the gridded contributor on the same inputs and compares every output. */
async function expectGridMatchesUnindexed(
  referencePoints: readonly Triple[],
  comparisonPoints: readonly Triple[],
  cores: readonly Triple[],
  normals: readonly Triple[],
  options: Options,
  cellSize: number
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const referenceView = fixture.view('float32x3', Float32Array.from(referencePoints.flat()));
    const comparisonView = fixture.view('float32x3', Float32Array.from(comparisonPoints.flat()));
    const coreView = fixture.view('float32x3', Float32Array.from(cores.flat()));
    const normalView = fixture.view('float32x3', Float32Array.from(normals.flat()));
    const transform = options.alignment ? fixture.view('float32', options.alignment) : undefined;
    // The grid has to contain the aligned comparison cloud, so it covers a margin of the translation.
    const grid = getGrid([...referencePoints, ...comparisonPoints], cellSize, 4 * cellSize);
    const outputs = [false, true].map(indexed => {
      const changes = fixture.view('float32x3', new Float32Array(cores.length * 3));
      const roughness = fixture.view('float32x3', new Float32Array(cores.length * 3));
      const counts = fixture.view('uint32', new Uint32Array(cores.length * 2));
      const status = fixture.view('uint32', new Uint32Array(cores.length));
      fixture.graph.add(
        new GPUPointCloudChange({
          id: `change-grid-equality-${indexed}`,
          reference: referenceView.view,
          comparison: comparisonView.view,
          cores: coreView.view,
          normals: normalView.view,
          alignment: transform ? {transform: transform.view} : 'already-aligned',
          radius: options.radius,
          halfDepth: options.halfDepth,
          minimumSupport: options.minimumSupport ?? 2,
          samplingCovariance: options.samplingCovariance,
          registrationError: options.registrationError,
          registrationCovariance: options.registrationCovariance,
          uncertaintyCombination: options.uncertaintyCombination,
          ...(indexed ? {grid} : {}),
          output: {
            changes: changes.view,
            roughness: roughness.view,
            counts: counts.view,
            status: status.view
          }
        })
      );
      return {changes, roughness, counts, status};
    });
    await fixture.run();
    const [unindexed, indexed] = outputs;
    // Counts and statuses are integers, so they must be identical.
    expect(await readIntegers(indexed.counts.buffer)).toEqual(
      await readIntegers(unindexed.counts.buffer)
    );
    const statuses = await readIntegers(unindexed.status.buffer);
    // The incomplete-grid flag is the only difference allowed: it marks cores whose cylinder leaves the grid.
    const indexedStatuses = await readIntegers(indexed.status.buffer);
    expect(
      indexedStatuses.map(status => status & ~GPU_POINT_CLOUD_CHANGE_STATUS_GRID_INCOMPLETE)
    ).toEqual(statuses);
    // Floating point sums visit the points in cell order, so allow float32 rounding.
    for (const name of ['changes', 'roughness'] as const) {
      const expected = await readFloats(unindexed[name].buffer);
      const actual = await readFloats(indexed[name].buffer);
      for (let element = 0; element < expected.length; element++) {
        expect(actual[element], `${name}[${element}]`).toBeCloseTo(expected[element], 4);
      }
    }
    return statuses;
  } finally {
    fixture.destroy();
  }
}

it('gridded M3C2 equals the unindexed scan on the pinned PDAL golden fixture', async () => {
  const rows = reference.cores.filter(row => !row.omitFirstReferencePoint);
  const statuses = await expectGridMatchesUnindexed(
    toTriples(reference.reference),
    toTriples(reference.comparison),
    rows.map(row => row.position as unknown as Triple),
    rows.map(row => row.normal as unknown as Triple),
    {radius: reference.radius, halfDepth: reference.halfDepth, minimumSupport: 2},
    reference.radius
  );
  expect(statuses).toEqual(new Array(rows.length).fill(0));
});

it('gridded M3C2 equals the unindexed scan on a random cloud with tilted normals, alignment and sampling covariance', async () => {
  const random = createRandom(20260504);
  const extent = 12;
  const surface = (x: number, y: number) => 0.4 * Math.sin(x * 0.7) + 0.3 * Math.cos(y * 0.5);
  const makeCloud = (count: number, lift: number) =>
    Array.from({length: count}, (): Triple => {
      const x = random() * extent;
      const y = random() * extent;
      return [x, y, surface(x, y) + lift + (random() - 0.5) * 0.05];
    });
  const referencePoints = makeCloud(3000, 0);
  const comparisonPoints = makeCloud(3000, 0.12);
  const cores: Triple[] = [];
  const normals: Triple[] = [];
  for (let core = 0; core < 120; core++) {
    const x = random() * extent;
    const y = random() * extent;
    cores.push([x, y, surface(x, y)]);
    const tilt = core % 3 === 0 ? 0 : (random() - 0.5) * 0.8;
    const length = Math.hypot(tilt, 1);
    normals.push([tilt / length, (tilt * 0.5) / length, 1 / length]);
  }
  // A core far outside every cloud has no support; an unusable core is invalid. Both must agree.
  cores.push([1e6, 1e6, 1e6], [Number.NaN, 0, 0]);
  normals.push([0, 0, 1], [0, 0, 1]);
  // Rotation of 0.05 radians about z and a small translation, column-major.
  const cosine = Math.cos(0.05);
  const sine = Math.sin(0.05);
  const alignment = new Float32Array([
    cosine,
    sine,
    0,
    0,
    -sine,
    cosine,
    0,
    0,
    0,
    0,
    1,
    0,
    0.2,
    -0.1,
    0,
    1
  ]);
  for (const options of [
    {radius: 1.2, halfDepth: 0.6},
    {
      radius: 1.2,
      halfDepth: 0.6,
      alignment,
      registrationError: 0.01,
      registrationCovariance: [1e-4, 0, 0, 1e-4, 0, 1e-4] as const,
      uncertaintyCombination: 'quadrature' as const,
      samplingCovariance: {
        referenceVariance: 0.01,
        comparisonVariance: 0.02,
        correlationLength: 0.5
      }
    }
  ]) {
    const statuses = await expectGridMatchesUnindexed(
      referencePoints,
      comparisonPoints,
      cores,
      normals,
      options,
      1
    );
    // The fixture exercises supported, unsupported and invalid cores.
    expect(statuses.slice(-2)).toEqual([2, 1]);
    expect(statuses.filter(status => status === 0).length).toBeGreaterThan(80);
  }
});

it('a core whose cylinder leaves the grid reports the incomplete-grid flag', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // Points fill x in [0, 10); the grid covers only x in [0, 5).
    const points = Array.from({length: 100}, (_, index): Triple => [index / 10, 0.5, 0]);
    const cloud = fixture.view('float32x3', Float32Array.from(points.flat()));
    const comparison = fixture.view('float32x3', Float32Array.from(points.flat()));
    const cores = fixture.view('float32x3', Float32Array.from([2, 0.5, 0, 9, 0.5, 0, 4.9, 0.5, 0]));
    const normals = fixture.view('float32x3', Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1]));
    const changes = fixture.view('float32x3', new Float32Array(9));
    const roughness = fixture.view('float32x3', new Float32Array(9));
    const counts = fixture.view('uint32', new Uint32Array(6));
    const status = fixture.view('uint32', new Uint32Array(3));
    fixture.graph.add(
      new GPUPointCloudChange({
        id: 'change-grid-incomplete',
        reference: cloud.view,
        comparison: comparison.view,
        cores: cores.view,
        normals: normals.view,
        alignment: 'already-aligned',
        radius: 0.5,
        halfDepth: 0.5,
        minimumSupport: 2,
        grid: {origin: [0, -2, -2], cellSize: 1, dimensions: [5, 4, 4]},
        output: {
          changes: changes.view,
          roughness: roughness.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    const values = await readIntegers(status.buffer);
    // Core 0 is interior, core 1 lies outside the grid, core 2 touches its upper edge.
    expect(values[0] & GPU_POINT_CLOUD_CHANGE_STATUS_GRID_INCOMPLETE).toBe(0);
    expect(values[1] & GPU_POINT_CLOUD_CHANGE_STATUS_GRID_INCOMPLETE).not.toBe(0);
    expect(values[2] & GPU_POINT_CLOUD_CHANGE_STATUS_GRID_INCOMPLETE).not.toBe(0);
  } finally {
    fixture.destroy();
  }
});
