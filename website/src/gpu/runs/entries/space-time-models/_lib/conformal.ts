// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared scene of the conformal prediction runs: Helene station rainfall (log of one plus the
// daily millimeters) predicted by the mean of the nearest stations of a training third. A calibration third supplies the held-out scores and a test third
// receives the intervals. The scale of each station is the spread of log rainfall among its nearest
// training stations, so the intervals can be wider where rainfall is more variable.

import type {Random, World} from '../../../inputs';
import {createRainScene, type RainScene} from './rain';
import type {Rectangle} from './surface';

/** Neighbours of a station used for its local spread. */
const SPREAD_NEIGHBOR_COUNT = 8;
const MINIMUM_SCALE = 0.2;

export type ConformalStations = {
  count: number;
  positions: Float32Array;
  /** Observed log(1 + rain) of the second day. */
  observations: Float32Array;
  /** Predictions: mean log rainfall of the nearest training stations. */
  predictions: Float32Array;
  /** Local spread of log rainfall, the heteroscedastic scale. */
  scales: Float32Array;
};

export type ConformalScene = {
  calibration: ConformalStations;
  test: ConformalStations;
  mapWidth: number;
  panel: Rectangle;
};

export function createConformalScene(
  random: Random,
  world: World,
  stationCount = 540
): ConformalScene {
  const rain: RainScene = createRainScene(random, world, stationCount);
  const usable = Array.from({length: rain.count}, (_, row) => row);
  const roles = usable.map((_, index) => index % 3);
  const trainingRows = usable.filter((_, index) => roles[index] === 0);
  const getLog = (row: number) => Math.log1p(rain.rain[row]);

  // The forecast is the mean log rainfall of the nearest training stations, and the scale their
  // spread, so both need only stations the forecast has not been calibrated on.
  const getNeighborStatistics = (row: number): {mean: number; spread: number} => {
    const x = rain.positions[row * 2];
    const y = rain.positions[row * 2 + 1];
    const nearest = trainingRows
      .filter(other => other !== row)
      .map(other => ({
        other,
        distance: Math.hypot(rain.positions[other * 2] - x, rain.positions[other * 2 + 1] - y)
      }))
      .sort((first, second) => first.distance - second.distance)
      .slice(0, SPREAD_NEIGHBOR_COUNT);
    const mean = nearest.reduce((sum, entry) => sum + getLog(entry.other), 0) / nearest.length;
    const spread = Math.sqrt(
      nearest.reduce((sum, entry) => sum + (getLog(entry.other) - mean) ** 2, 0) / nearest.length
    );
    return {mean, spread};
  };

  const collect = (role: number): ConformalStations => {
    const rows = usable.filter((_, index) => roles[index] === role);
    const positions = new Float32Array(rows.length * 2);
    rows.forEach((row, index) =>
      positions.set([rain.positions[row * 2], rain.positions[row * 2 + 1]], index * 2)
    );
    return {
      count: rows.length,
      positions,
      observations: Float32Array.from(rows, getLog),
      predictions: Float32Array.from(rows, row => getNeighborStatistics(row).mean),
      scales: Float32Array.from(rows, row =>
        Math.max(getNeighborStatistics(row).spread, MINIMUM_SCALE)
      )
    };
  };

  return {
    calibration: collect(1),
    test: collect(2),
    mapWidth: rain.mapWidth,
    panel: rain.panel
  };
}

/** Share of test observations inside their intervals (unbounded sides cover) and the mean width of the bounded ones. */
export function summarizeIntervals(
  intervals: ArrayLike<number>,
  observations: ArrayLike<number>
): {coverage: number; meanWidth: number; boundedShare: number} {
  let covered = 0;
  let widthSum = 0;
  let bounded = 0;
  let valid = 0;
  for (let row = 0; row < observations.length; row++) {
    const lower = intervals[row * 2];
    const upper = intervals[row * 2 + 1];
    if (Number.isNaN(lower) || Number.isNaN(upper)) continue;
    valid++;
    if (observations[row] >= lower && observations[row] <= upper) covered++;
    if (Number.isFinite(lower) && Number.isFinite(upper)) {
      widthSum += upper - lower;
      bounded++;
    }
  }
  return {
    coverage: valid > 0 ? covered / valid : Number.NaN,
    meanWidth: bounded > 0 ? widthSum / bounded : Number.NaN,
    boundedShare: valid > 0 ? bounded / valid : 0
  };
}

/** Interval widths per row; NaN where a side is unbounded or the interval is invalid. */
export function getIntervalWidths(intervals: ArrayLike<number>, rowCount: number): Float32Array {
  return Float32Array.from({length: rowCount}, (_, row) => {
    const width = intervals[row * 2 + 1] - intervals[row * 2];
    return Number.isFinite(width) ? width : Number.NaN;
  });
}

/** Millimeters of rain of a log(1 + rain) value. */
export function toMillimeters(logValue: number): number {
  return Math.max(0, Math.expm1(logValue));
}
