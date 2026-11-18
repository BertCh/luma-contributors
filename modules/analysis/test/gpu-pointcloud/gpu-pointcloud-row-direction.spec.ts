// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUPointCloudGround,
  GPUPointCloudHeight,
  GPUPointCloudSurfaces
} from '../../src/pointcloud-analysis/index';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

import {
  arrangeRasterRows,
  POINT_CLOUD_ROW_DIRECTIONS,
  withPointCloudRowDirection
} from './row-direction-fixtures';

const WIDTH = 3;
const HEIGHT = 3;

// Point-cloud rasters default to `rowDirection: 'south'`, like the raster and terrain families: row 0
// is the top row at the maximum y. `'north'` (row 0 at the minimum y) is the explicit alternative.
// Every case below writes its raster with row 0 at the minimum y and lays it out for the direction
// under test, so the same input must give the same per-point answer, and a raster output the same
// cells with the rows mirrored.

it('Surfaces: south (the default) is the row mirror of north', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const points: number[] = [];
    // One point at every cell center, with an elevation that names the cell (north row * 10 + column + 1).
    for (let row = 0; row < HEIGHT; row++) {
      for (let column = 0; column < WIDTH; column++) {
        points.push(column + 0.5, row + 0.5, row * 10 + column + 1);
      }
    }
    // A point exactly on a cell corner belongs to the cell above and to the right in every direction.
    points.push(1, 2, 99);
    // The top and right edges are outside the half-open grid.
    points.push(1.5, 3, 77, 3, 1.5, 78);
    const positions = fixture.view('float32x3', new Float32Array(points));
    const pointCount = points.length / 3;
    const classifications = fixture.view('uint32', new Uint32Array(pointCount).fill(2));
    const runs = POINT_CLOUD_ROW_DIRECTIONS.map(({label, rowDirection}) => {
      const outputs = {
        terrain: fixture.view('float32', new Float32Array(WIDTH * HEIGHT)),
        surface: fixture.view('float32', new Float32Array(WIDTH * HEIGHT)),
        canopy: fixture.view('float32', new Float32Array(WIDTH * HEIGHT)),
        coverage: fixture.view('uint32', new Uint32Array(WIDTH * HEIGHT))
      };
      fixture.graph.add(
        new GPUPointCloudSurfaces({
          id: `surfaces-${label}`,
          positions: positions.view,
          classifications: classifications.view,
          grid: withPointCloudRowDirection(
            {origin: [0, 0] as [number, number], cellSize: 1, width: WIDTH, height: HEIGHT},
            rowDirection
          ),
          output: {
            terrain: outputs.terrain.view,
            surface: outputs.surface.view,
            canopy: outputs.canopy.view,
            coverage: outputs.coverage.view
          }
        })
      );
      return {label, rowDirection, outputs};
    });
    await fixture.run();
    const northSurface = [1, 2, 3, 11, 12, 13, 21, 99, 23];
    for (const {label, rowDirection, outputs} of runs) {
      // North: row 0 is the bottom row; the corner point (1, 2) lands in north row 2, column 1.
      // South and the default: row 0 is the top row, so the same cells appear with the rows reversed.
      const expected = arrangeRasterRows(northSurface, WIDTH, HEIGHT, rowDirection);
      expect(await readFloats(outputs.surface.buffer), label).toEqual(expected);
      // Terrain is the minimum ground elevation of the cell, so the corner point 99 does not lift it.
      expect(await readFloats(outputs.terrain.buffer), label).toEqual(
        arrangeRasterRows([1, 2, 3, 11, 12, 13, 21, 22, 23], WIDTH, HEIGHT, rowDirection)
      );
      expect(await readIntegers(outputs.coverage.buffer), label).toEqual(
        arrangeRasterRows(Array(WIDTH * HEIGHT).fill(3), WIDTH, HEIGHT, rowDirection)
      );
    }
    expect(await readFloats(runs[1].outputs.surface.buffer)).toEqual([
      21, 99, 23, 11, 12, 13, 1, 2, 3
    ]);
    expect(await readFloats(runs[2].outputs.surface.buffer)).toEqual(
      await readFloats(runs[1].outputs.surface.buffer)
    );
  } finally {
    fixture.destroy();
  }
});

// Terrain 1 3 / 7 9 (north rows), slope 2 in x and 6 in y. Points, with elevation 10:
const HEIGHT_POINTS = [
  [0.5, 0.5], // center of the cell holding 1
  [1.5, 0.5], // 3
  [0.5, 1.5], // 7
  [1.5, 1.5], // 9
  [1, 1], // grid center: the cell above and to the right (9) for cell sampling
  [0.75, 0.75],
  [0.75, 1.25],
  [3, 0.5] // right edge: outside
] as const;
const NORTH_TERRAIN = [1, 3, 7, 9];

/** Bilinear elevation at (x, y) of the 2 by 2 north raster, all four corners in range. */
function interpolateBilinear(x: number, y: number): number {
  const fractionX = x - 0.5;
  const fractionY = y - 0.5;
  const bottom = NORTH_TERRAIN[0] + fractionX * (NORTH_TERRAIN[1] - NORTH_TERRAIN[0]);
  const top = NORTH_TERRAIN[2] + fractionX * (NORTH_TERRAIN[3] - NORTH_TERRAIN[2]);
  return bottom + fractionY * (top - bottom);
}

/** Inverse-distance (power 2, radius 2) elevation of the four cell centers, as the contributor defines it. */
function interpolateIDW(x: number, y: number): number {
  const centers = NORTH_TERRAIN.map((value, index) => ({
    value,
    distanceSquared: (x - ((index % 2) + 0.5)) ** 2 + (y - (Math.floor(index / 2) + 0.5)) ** 2
  })).filter(({distanceSquared}) => distanceSquared <= 4);
  const exact = centers.find(({distanceSquared}) => distanceSquared === 0);
  if (exact) {
    return exact.value;
  }
  const weightSum = centers.reduce((sum, {distanceSquared}) => sum + 1 / distanceSquared, 0);
  return (
    centers.reduce((sum, {value, distanceSquared}) => sum + value / distanceSquared, 0) / weightSum
  );
}

const HEIGHT_MODES = [
  {
    method: 'cell',
    interpolation: undefined,
    // The cell holding each point: 1, 3, 7, 9, 9 (corner), 1, 7, then outside.
    expected: [9, 7, 3, 1, 1, 9, 3, 0],
    status: [0, 0, 0, 0, 0, 0, 0, 1]
  },
  {
    method: 'bilinear',
    interpolation: {method: 'bilinear'} as const,
    // Centers reproduce the cell; (1, 1) is the mean 5; (0.75, 0.75) is 3 and (0.75, 1.25) is 6.
    expected: [9, 7, 3, 1, 5, 7, 4, 0],
    status: [0, 0, 0, 0, 0, 0, 0, 1]
  },
  {
    method: 'idw',
    interpolation: {method: 'idw', radius: 2, power: 2} as const,
    expected: HEIGHT_POINTS.map(([x, y]) => 10 - interpolateIDW(x, y)).map((value, index) =>
      index === 7 ? 0 : value
    ),
    status: [0, 0, 0, 0, 0, 0, 0, 1]
  }
];

for (const {method, interpolation, expected, status: expectedStatus} of HEIGHT_MODES) {
  for (const {label, rowDirection} of POINT_CLOUD_ROW_DIRECTIONS) {
    it(`Height ${method}: the same cells give the same heights, rows ${label}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        const positions = fixture.view(
          'float32x3',
          new Float32Array(HEIGHT_POINTS.flatMap(([x, y]) => [x, y, 10]))
        );
        const terrain = fixture.view(
          'float32',
          new Float32Array(arrangeRasterRows(NORTH_TERRAIN, 2, 2, rowDirection))
        );
        const coverage = fixture.view('uint32', new Uint32Array(4).fill(1));
        const heights = fixture.view('float32', new Float32Array(HEIGHT_POINTS.length));
        const status = fixture.view('uint32', new Uint32Array(HEIGHT_POINTS.length));
        fixture.graph.add(
          new GPUPointCloudHeight({
            positions: positions.view,
            grid: withPointCloudRowDirection(
              {origin: [0, 0] as [number, number], cellSize: 1, width: 2, height: 2},
              rowDirection
            ),
            terrain: terrain.view,
            coverage: coverage.view,
            interpolation,
            output: {heights: heights.view, status: status.view}
          })
        );
        await fixture.run();
        expect(await readIntegers(status.buffer)).toEqual(expectedStatus);
        const actual = await readFloats(heights.buffer);
        expected.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, 4));
        if (method === 'bilinear') {
          // The hand-worked numbers agree with an independent formula.
          HEIGHT_POINTS.slice(0, 7).forEach(([x, y], index) =>
            expect(actual[index]).toBeCloseTo(10 - interpolateBilinear(x, y), 4)
          );
        }
      } finally {
        fixture.destroy();
      }
    });
  }
}

for (const {label, rowDirection} of POINT_CLOUD_ROW_DIRECTIONS) {
  it(`Ground: coverage and openings follow the grid rows, rows ${label}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const size = 4;
      // North rows 0-2 are covered, row 3 is not; a spike of 5 sits in north cell (2, 2).
      const surface = Array.from({length: size * size}, (_, cell) =>
        cell === 2 * size + 2 ? 5 : 0
      );
      const covered = Array.from({length: size * size}, (_, cell) =>
        Math.floor(cell / size) < 3 ? 1 : 0
      );
      const positions = new Float32Array(size * size * 3);
      surface.forEach((value, cell) => {
        positions.set([(cell % size) + 0.5, Math.floor(cell / size) + 0.5, value], cell * 3);
      });
      const groundMask = fixture.view('uint32', new Uint32Array(size * size));
      const status = fixture.view('uint32', new Uint32Array(size * size));
      fixture.graph.add(
        new GPUPointCloudGround({
          positions: fixture.view('float32x3', positions).view,
          classifications: fixture.view('uint32', new Uint32Array(size * size).fill(1)).view,
          grid: withPointCloudRowDirection(
            {origin: [0, 0] as [number, number], cellSize: 1, width: size, height: size},
            rowDirection
          ),
          radii: [1],
          initialThreshold: 0.3,
          slope: 0,
          maximumThreshold: 1,
          minimumSurface: fixture.view(
            'float32',
            new Float32Array(arrangeRasterRows(surface, size, size, rowDirection))
          ).view,
          coverage: fixture.view(
            'uint32',
            new Uint32Array(arrangeRasterRows(covered, size, size, rowDirection))
          ).view,
          output: {
            groundMask: groundMask.view,
            outputClassifications: fixture.view('uint32', new Uint32Array(size * size)).view,
            status: status.view
          }
        })
      );
      await fixture.run();
      // Per point (north cell order): ground unless the spike (opened away, residual 5) or the
      // uncovered north row 3 (status bit 1, not ground).
      expect(await readIntegers(groundMask.buffer)).toEqual(
        covered.map((isCovered, cell) => (isCovered && cell !== 2 * size + 2 ? 1 : 0))
      );
      const actualStatus = await readIntegers(status.buffer);
      covered.forEach((isCovered, cell) => {
        expect(actualStatus[cell] & 1).toBe(isCovered ? 0 : 1);
      });
    } finally {
      fixture.destroy();
    }
  });
}
