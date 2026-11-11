// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Hurricane Helene station rainfall over the Carolinas (two observation days) for the
// space-time-models runs: the map sits in the left part of the picture, a panel on the right.

import {getRainStations} from '../../../../data/point-observations';
import type {Random, World} from '../../../inputs';
import type {Rectangle} from './surface';

/** Share of the picture width the station map takes. */
export const MAP_SHARE = 0.44;

export type RainScene = {
  count: number;
  positions: Float32Array;
  /** Measured rain of the second day, millimeters. */
  rain: Float32Array;
  /** Measured rain of the first day, millimeters (NaN where missing). */
  previousRain: Float32Array;
  /** log(1 + rain) minus a planar trend fitted by least squares: a zero-mean residual. */
  residuals: Float32Array;
  /** log(1 + previousRain) minus a planar trend fitted to day one. */
  previousResiduals: Float32Array;
  mapWidth: number;
  panel: Rectangle;
};

export function createRainScene(random: Random, world: World, count: number): RainScene {
  const mapWidth = world.width * MAP_SHARE;
  const stations = getRainStations(random, {width: mapWidth, height: world.height}, count, true);
  const logRain = Array.from(stations.rain, value => Math.log1p(value));
  const trend = fitPlanarTrend(stations.positions, logRain);
  // Each day gets its own trend, so both days' residuals are mean zero (simple kriging, Vecchia).
  const previousLogRain = Array.from(stations.previousRain, value =>
    Number.isFinite(value) ? Math.log1p(value) : Number.NaN
  );
  const previousTrend = fitPlanarTrend(stations.positions, previousLogRain);
  return {
    count,
    positions: stations.positions,
    rain: stations.rain,
    previousRain: stations.previousRain,
    residuals: Float32Array.from(logRain, (value, row) => value - trend(row)),
    previousResiduals: Float32Array.from(previousLogRain, (value, row) =>
      Number.isFinite(value) ? value - previousTrend(row) : Number.NaN
    ),
    mapWidth,
    panel: {
      x0: mapWidth + world.width * 0.07,
      y0: 0.1,
      x1: world.width * 0.97,
      y1: 0.92
    }
  };
}

/** Rows `[value, observationErrorVariance, reserved, validity]` of the likelihood operations. */
export function packObservations(values: ArrayLike<number>): Float32Array {
  const rows = new Float32Array(values.length * 4);
  for (let row = 0; row < values.length; row++) {
    const isValid = Number.isFinite(values[row]);
    rows[row * 4] = isValid ? values[row] : 0;
    rows[row * 4 + 3] = isValid ? 1 : 0;
  }
  return rows;
}

/** One segment from each row to each row it conditions on, for drawing a CSR of neighborhoods. */
export function createNeighborhoodSegments(
  positions: ArrayLike<number>,
  offsets: ArrayLike<number>,
  neighbors: ArrayLike<number>
): Float32Array {
  const segments = new Float32Array(Math.max(neighbors.length, 1) * 4);
  for (let row = 0; row + 1 < offsets.length; row++) {
    for (let entry = offsets[row]; entry < offsets[row + 1]; entry++) {
      const other = neighbors[entry];
      segments.set(
        [
          positions[row * 2],
          positions[row * 2 + 1],
          positions[other * 2],
          positions[other * 2 + 1]
        ],
        entry * 4
      );
    }
  }
  return segments;
}

/**
 * Conditioning links of only the `count` stations with the lowest finite terms (the stations the
 * current covariance surprises most); drawing every link makes an unreadable tangle.
 */
export function createWorstStationSegments(
  allSegments: Float32Array,
  offsets: ArrayLike<number>,
  terms: ArrayLike<number>,
  count: number
): Float32Array {
  const rows = Array.from({length: offsets.length - 1}, (_, row) => row)
    .filter(row => Number.isFinite(terms[row]))
    .sort((left, right) => terms[left] - terms[right])
    .slice(0, count);
  const selected: number[] = [];
  for (const row of rows) {
    for (let entry = offsets[row]; entry < offsets[row + 1]; entry++) {
      selected.push(...allSegments.subarray(entry * 4, entry * 4 + 4));
    }
  }
  return selected.length > 0 ? Float32Array.from(selected) : new Float32Array(4);
}

/** Number of non-finite cells of a swept surface (evaluations where some site failed). */
export function countFailedCells(values: ArrayLike<number>): number {
  let failed = 0;
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index])) failed++;
  }
  return failed;
}

/** Least-squares fit of `a + b x + c y`; returns the fitted value per row. */
function fitPlanarTrend(positions: Float32Array, values: number[]): (row: number) => number {
  const normal = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  const right = [0, 0, 0];
  values.forEach((value, row) => {
    if (!Number.isFinite(value)) return;
    const design = [1, positions[row * 2], positions[row * 2 + 1]];
    for (let first = 0; first < 3; first++) {
      right[first] += design[first] * value;
      for (let second = 0; second < 3; second++)
        normal[first * 3 + second] += design[first] * design[second];
    }
  });
  const coefficients = solveThree(normal, right);
  return row =>
    coefficients[0] +
    coefficients[1] * positions[row * 2] +
    coefficients[2] * positions[row * 2 + 1];
}

/** Cramer's rule for a 3 x 3 system. */
export function solveThree(matrix: number[], right: number[]): number[] {
  const determinant = (entries: number[]) =>
    entries[0] * (entries[4] * entries[8] - entries[5] * entries[7]) -
    entries[1] * (entries[3] * entries[8] - entries[5] * entries[6]) +
    entries[2] * (entries[3] * entries[7] - entries[4] * entries[6]);
  const whole = determinant(matrix);
  return [0, 1, 2].map(column => {
    const replaced = [...matrix];
    for (let row = 0; row < 3; row++) replaced[row * 3 + column] = right[row];
    return determinant(replaced) / whole;
  });
}
