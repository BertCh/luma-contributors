// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerLagrangianField} from '@luma-contributors/kepler';
import {sampleFlowFieldGrid, type FlowFieldAnalysisSnapshot} from './flow-field-grid';

const MERCATOR_RADIUS = 6378137;
const RADIANS_PER_DEGREE = Math.PI / 180;

export function projectFlowPosition(longitude: number, latitude: number): [number, number] {
  return [
    MERCATOR_RADIUS * longitude * RADIANS_PER_DEGREE,
    MERCATOR_RADIUS * Math.log(Math.tan(Math.PI / 4 + (latitude * RADIANS_PER_DEGREE) / 2))
  ];
}
export function unprojectFlowPosition(x: number, y: number): [number, number] {
  return [
    x / MERCATOR_RADIUS / RADIANS_PER_DEGREE,
    (2 * Math.atan(Math.exp(y / MERCATOR_RADIUS)) - Math.PI / 2) / RADIANS_PER_DEGREE
  ];
}

/**
 * Resamples the particle workflow's endpoint-spaced degree grid onto regular Mercator centres.
 * Ground east/north m/s becomes projected m/s by sec(latitude). Subtracting the southwest
 * origin keeps integration Float32 detail at continental offsets. Invalid corners remain NaN.
 * This is a frozen, physical-velocity snapshot, independent of the particle visual speed scale.
 */
export function prepareFlowFieldLagrangian(grid: FlowFieldAnalysisSnapshot): {
  field: KeplerLagrangianField;
  origin: [number, number];
} {
  if (grid.mode === 'ELEVATION')
    throw new Error(
      'Elevation flow grids contain dimensionless slopes, not physical velocities; use UV or SPEED_DIR in m/s for diagnostics'
    );
  if (grid.mode !== 'UV' && grid.mode !== 'SPEED_DIR')
    throw new Error('Physical diagnostics require an explicit UV or SPEED_DIR velocity field mode');
  if (
    grid.columns < 4 ||
    grid.rows < 4 ||
    grid.columns > 512 ||
    grid.rows > 512 ||
    grid.texels.length !== grid.columns * grid.rows * 4 ||
    !grid.bounds.every(Number.isFinite) ||
    grid.bounds[0] < -180 ||
    grid.bounds[2] > 180 ||
    grid.bounds[1] < -85 ||
    grid.bounds[3] > 85 ||
    grid.bounds[2] <= grid.bounds[0] ||
    grid.bounds[3] <= grid.bounds[1]
  )
    throw new Error(
      'Diagnostics need a finite flow grid within ±85° latitude and a non-wrapping extent'
    );
  const origin = projectFlowPosition(grid.bounds[0], grid.bounds[1]);
  const northeast = projectFlowPosition(grid.bounds[2], grid.bounds[3]);
  const spacingX = (northeast[0] - origin[0]) / grid.columns;
  const spacingY = (northeast[1] - origin[1]) / grid.rows;
  const velocitiesStart = new Float32Array(grid.columns * grid.rows * 2).fill(NaN);
  for (let row = 0; row < grid.rows; row++)
    for (let column = 0; column < grid.columns; column++) {
      const [longitude, latitude] = unprojectFlowPosition(
        origin[0] + (column + 0.5) * spacingX,
        origin[1] + (row + 0.5) * spacingY
      );
      const velocity = sampleFlowFieldGrid(grid, longitude, latitude);
      if (!velocity) continue;
      const scale = 1 / Math.cos(latitude * RADIANS_PER_DEGREE);
      const index = row * grid.columns + column;
      velocitiesStart[index * 2] = velocity.east * scale;
      velocitiesStart[index * 2 + 1] = velocity.north * scale;
    }
  return {
    origin,
    field: {
      width: grid.columns,
      height: grid.rows,
      extent: [0, 0, spacingX, spacingY],
      velocitiesStart
    }
  };
}
