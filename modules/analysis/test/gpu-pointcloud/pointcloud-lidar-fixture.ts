// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  LocalMetricProjection,
  getBboxCenter,
  projectLngLatArray
} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/projection';
import {getPointCloudRowDirection} from '@luma-contributors/analysis/pointcloud-analysis';

/** Use the scene's exact projection and float32 upload convention for the bundled USGS rows. */
export function projectLidarGroundTile(
  longitudeLatitude: Float64Array,
  elevation: Float32Array,
  bounds: readonly [number, number, number, number]
) {
  const projected = projectLngLatArray(
    new LocalMetricProjection(getBboxCenter(bounds)),
    longitudeLatitude
  );
  const positions = new Float32Array(elevation.length * 3);
  for (let row = 0; row < elevation.length; row++)
    positions.set([projected[row * 2], projected[row * 2 + 1], elevation[row]], row * 3);
  return positions;
}

/** Independent bounded CPU replay of seed minima, square opening, and final support decisions. */
export function replayLidarGroundSupport(
  positions: Float32Array,
  classes: Uint32Array,
  grid: {
    origin: readonly [number, number];
    width: number;
    height: number;
    cellSize: number;
    rowDirection?: 'north' | 'south';
  },
  radii: readonly number[]
) {
  const cells = grid.width * grid.height;
  const pointCells = Array.from({length: classes.length}, (_, row) => {
    const column = Math.floor(
      Math.fround(Math.fround(positions[row * 3] - grid.origin[0]) / grid.cellSize)
    );
    const vertical = Math.floor(
      Math.fround(Math.fround(positions[row * 3 + 1] - grid.origin[1]) / grid.cellSize)
    );
    const rasterRow =
      getPointCloudRowDirection(grid) === 'north' ? vertical : grid.height - 1 - vertical;
    return column >= 0 && column < grid.width && vertical >= 0 && vertical < grid.height
      ? rasterRow * grid.width + column
      : -1;
  });
  const coverage = new Uint32Array(cells);
  const minimum = new Float32Array(cells).fill(Infinity);
  for (let row = 0; row < classes.length; row++) {
    const cell = pointCells[row];
    if (cell < 0 || classes[row] !== 2) continue;
    coverage[cell] = 1;
    minimum[cell] = Math.min(minimum[cell], positions[row * 3 + 2]);
  }
  const mask = new Uint32Array(classes.length).fill(1);
  const status = new Uint32Array(classes.length);
  let previousRadius = 0;
  for (const radius of radii) {
    const morphology = (source: Float32Array, erode: boolean, partial?: Uint32Array) => {
      const values = new Float32Array(cells);
      const support = new Uint32Array(cells);
      for (let cell = 0; cell < cells; cell++) {
        const column = cell % grid.width;
        const row = Math.floor(cell / grid.width);
        let value = erode ? Infinity : -Infinity;
        let count = 0;
        let complete = true;
        for (let vertical = -radius; vertical <= radius; vertical++)
          for (let horizontal = -radius; horizontal <= radius; horizontal++) {
            const otherColumn = column + horizontal;
            const otherRow = row + vertical;
            if (
              otherColumn < 0 ||
              otherColumn >= grid.width ||
              otherRow < 0 ||
              otherRow >= grid.height
            )
              continue;
            const other = otherRow * grid.width + otherColumn;
            if (!coverage[other]) continue;
            value = erode ? Math.min(value, source[other]) : Math.max(value, source[other]);
            count++;
            if (partial && partial[other] !== 0) complete = false;
          }
        values[cell] = count ? value : 0;
        support[cell] =
          (count === (2 * radius + 1) ** 2 && complete ? 0 : 2) | (coverage[cell] ? 0 : 4);
      }
      return {values, support};
    };
    const erosion = morphology(minimum, true);
    const opening = morphology(erosion.values, false, erosion.support);
    const threshold = Math.fround(
      Math.min(2, 0.3 + 0.1 * grid.cellSize * (radius - previousRadius))
    );
    for (let row = 0; row < classes.length; row++) {
      const cell = pointCells[row];
      const valid = cell >= 0 && (opening.support[cell] & 4) === 0;
      mask[row] = Number(
        valid &&
          mask[row] !== 0 &&
          Math.fround(positions[row * 3 + 2] - opening.values[cell]) <= threshold
      );
      status[row] = valid ? opening.support[cell] & 2 : 1;
    }
    previousRadius = radius;
  }
  const supportedCells = new Set(
    pointCells.filter((cell, row) => mask[row] !== 0 && status[row] === 0)
  );
  return {
    rawAccepted: mask.reduce((sum, value) => sum + value, 0),
    supportedAccepted: mask.reduce(
      (sum, value, row) => sum + Number(value !== 0 && status[row] === 0),
      0
    ),
    supportedHeights: pointCells.filter(cell => supportedCells.has(cell)).length
  };
}
