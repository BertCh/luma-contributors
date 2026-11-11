// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A point belongs to the cell that contains it under the half-open rule: with o = (y - origin.y) /
// cellSize, it is inside when 0 <= o < height, in north row floor(o), which is south row
// height - 1 - floor(o). Before 2026-11-11 the south rows were computed as ceil(height - o) - 1 in
// float32, and height - o rounds up when o is just below a small integer: with height 1000 a point
// at y = 0.99999994 landed in south row 998 instead of 999, and a point just below the origin was
// inside. The reference below is binary64 arithmetic on the float32 coordinates (origin 0 and cell
// size 1, so o is the stored y exactly).

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_POINT_CLOUD_HEIGHT_STATUS_INVALID_POINT,
  GPU_POINT_CLOUD_HEIGHT_STATUS_VALID,
  GPUPointCloudHeight,
  GPUPointCloudSurfaces
} from '../../src/pointcloud-analysis/index';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {arrangeRasterRows, POINT_CLOUD_ROW_DIRECTIONS} from './row-direction-fixtures';

const ROWS = 1000;
const RANDOM_POINT_COUNT = 200_000;

const BITS = new DataView(new ArrayBuffer(4));
/** The float32 next to a positive float32 `value`, toward `+1` (up) or `-1` (down). */
function stepFloat32(value: number, direction: 1 | -1): number {
  BITS.setFloat32(0, value);
  BITS.setUint32(0, BITS.getUint32(0) + direction);
  return BITS.getFloat32(0);
}

/** Binary64 reference: the row of the cell holding float32 y, or -1 outside the grid. */
function getReferenceRow(y: number, rowDirection: 'north' | 'south' | undefined): number {
  if (!(y >= 0 && y < ROWS)) {
    return -1;
  }
  const northRow = Math.floor(y);
  return rowDirection === 'north' ? northRow : ROWS - 1 - northRow;
}

/** Every row boundary, and the float32 just above and just below it, plus points off each edge. */
function getBoundaryYs(): number[] {
  const values = [-(2 ** -20), -1e-6, -0.5, ROWS, stepFloat32(ROWS, 1), ROWS + 0.5];
  values.push(2 ** -20);
  for (let boundary = 1; boundary <= ROWS; boundary++) {
    values.push(boundary, stepFloat32(boundary, -1), stepFloat32(boundary, 1));
  }
  values.push(0);
  return values.map(Math.fround);
}

/** Uniform float32 y over the grid and one row beyond each edge, from a fixed LCG. */
function getRandomYs(): number[] {
  let state = 0x2545f491;
  const values: number[] = [];
  for (let index = 0; index < RANDOM_POINT_COUNT; index++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values.push(Math.fround((state / 2 ** 32) * (ROWS + 2) - 1));
  }
  return values;
}

/**
 * Runs GPUPointCloudHeight on a one-column grid whose terrain cell `i` holds `i + 1`, with every
 * point at elevation 0, so a valid height is minus one more than the cell index of the point.
 */
async function runHeights(
  ys: number[],
  rowDirection: 'north' | 'south' | undefined,
  interpolation?: {method: 'bilinear'} | {method: 'idw'; radius: number}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = new Float32Array(ys.length * 3);
    ys.forEach((y, index) => {
      positions[index * 3] = 0.5;
      positions[index * 3 + 1] = y;
    });
    const heights = fixture.view('float32', new Float32Array(ys.length));
    const status = fixture.view('uint32', new Uint32Array(ys.length));
    fixture.graph.add(
      new GPUPointCloudHeight({
        positions: fixture.view('float32x3', positions).view,
        grid: {
          origin: [0, 0],
          cellSize: 1,
          width: 1,
          height: ROWS,
          ...(rowDirection ? {rowDirection} : {})
        },
        terrain: fixture.view(
          'float32',
          Float32Array.from({length: ROWS}, (_, cell) => cell + 1)
        ).view,
        coverage: fixture.view('uint32', new Uint32Array(ROWS).fill(1)).view,
        ...(interpolation ? {interpolation} : {}),
        heights: heights.view,
        status: status.view
      })
    );
    await fixture.run();
    return {
      heights: await readFloats(heights.buffer),
      status: await readIntegers(status.buffer)
    };
  } finally {
    fixture.destroy();
  }
}

/** Cell sampling: the row of every point, or -1 when its status says it is outside. */
async function getHeightRows(ys: number[], rowDirection: 'north' | 'south' | undefined) {
  const {heights, status} = await runHeights(ys, rowDirection);
  return ys.map((_, index) => {
    if (status[index] === GPU_POINT_CLOUD_HEIGHT_STATUS_INVALID_POINT) {
      return -1;
    }
    expect(status[index]).toBe(GPU_POINT_CLOUD_HEIGHT_STATUS_VALID);
    return -heights[index] - 1;
  });
}

function getMismatches(ys: number[], rows: number[], rowDirection: 'north' | 'south' | undefined) {
  return ys
    .map((y, index) => ({y, row: rows[index], expected: getReferenceRow(y, rowDirection)}))
    .filter(({row, expected}) => row !== expected);
}

it('Height (cell) assigns points at and next to every row boundary to their cell', async () => {
  const ys = getBoundaryYs();
  for (const {label, rowDirection} of POINT_CLOUD_ROW_DIRECTIONS) {
    const rows = await getHeightRows(ys, rowDirection);
    expect(getMismatches(ys, rows, rowDirection), label).toEqual([]);
  }
});

it('Height (bilinear and IDW) treats the grid edges as half-open', async () => {
  const ys = getBoundaryYs();
  for (const {label, rowDirection} of POINT_CLOUD_ROW_DIRECTIONS) {
    for (const interpolation of [
      {method: 'bilinear'} as const,
      {method: 'idw', radius: 2} as const
    ]) {
      const {status} = await runHeights(ys, rowDirection, interpolation);
      const mismatches = ys.filter(
        (y, index) =>
          (status[index] === GPU_POINT_CLOUD_HEIGHT_STATUS_INVALID_POINT) !==
          (getReferenceRow(y, rowDirection) === -1)
      );
      expect(mismatches, `${label} ${interpolation.method}`).toEqual([]);
    }
  }
});

it('Height (cell) keeps the cell of every random point', async () => {
  const ys = getRandomYs();
  for (const {label, rowDirection} of POINT_CLOUD_ROW_DIRECTIONS) {
    const rows = await getHeightRows(ys, rowDirection);
    expect(getMismatches(ys, rows, rowDirection), label).toEqual([]);
  }
});

it('Surfaces (and its grid index) put both ends of every row in that row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // North row k holds y = k with elevation -(k + 1) and the float32 just below k + 1 with
    // elevation k + 1: one is the minimum and the other the maximum of the cell, so losing either
    // point, or gaining one from a neighbour, changes the terrain or the surface of a cell.
    const points: number[] = [];
    for (let row = 0; row < ROWS; row++) {
      points.push(0.5, row, -(row + 1), 0.5, stepFloat32(row + 1, -1), row + 1);
    }
    const positions = fixture.view('float32x3', new Float32Array(points));
    const classifications = fixture.view('uint32', new Uint32Array(2 * ROWS).fill(2));
    const runs = POINT_CLOUD_ROW_DIRECTIONS.map(({label, rowDirection}) => {
      const outputs = {
        terrain: fixture.view('float32', new Float32Array(ROWS)),
        surface: fixture.view('float32', new Float32Array(ROWS)),
        canopy: fixture.view('float32', new Float32Array(ROWS)),
        coverage: fixture.view('uint32', new Uint32Array(ROWS))
      };
      fixture.graph.add(
        new GPUPointCloudSurfaces({
          id: `surfaces-${label}`,
          positions: positions.view,
          classifications: classifications.view,
          grid: {
            origin: [0, 0],
            cellSize: 1,
            width: 1,
            height: ROWS,
            ...(rowDirection ? {rowDirection} : {})
          },
          terrain: outputs.terrain.view,
          surface: outputs.surface.view,
          canopy: outputs.canopy.view,
          coverage: outputs.coverage.view
        })
      );
      return {label, rowDirection, outputs};
    });
    await fixture.run();
    const northSurface = Array.from({length: ROWS}, (_, row) => row + 1);
    for (const {label, rowDirection, outputs} of runs) {
      expect(await readFloats(outputs.surface.buffer), label).toEqual(
        arrangeRasterRows(northSurface, 1, ROWS, rowDirection)
      );
      expect(await readFloats(outputs.terrain.buffer), label).toEqual(
        arrangeRasterRows(
          northSurface.map(value => -value),
          1,
          ROWS,
          rowDirection
        )
      );
    }
  } finally {
    fixture.destroy();
  }
});
