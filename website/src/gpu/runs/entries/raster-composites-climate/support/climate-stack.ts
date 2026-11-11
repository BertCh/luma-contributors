// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Scenario for the climate operations: a synthetic thirty-year monthly precipitation record on a
// small grid over the Carolinas and a synthetic daily year. The checked-in data holds no climate
// record, so only the spatial pattern is real: NOAA GHCN-Daily station rainfall of September 27,
// 2024 (Hurricane Helene), interpolated to the grid, sets how wet each cell is. Monthly totals are
// gamma distributed around a seasonal mean that peaks in September. In the final year, the
// September total gets the interpolated Helene rainfall on top, so the observed footprint is the
// anomaly the operations should find.

import {getRainStations} from '../../../../data/point-observations';
import {createSmoothField, type Random, type World} from '../../../inputs';

export const YEAR_COUNT = 30;
export const MONTHS_PER_YEAR = 12;
export const MONTH_COUNT = YEAR_COUNT * MONTHS_PER_YEAR;
export const HELENE_MONTH = 8;
export const MONTH_NAMES = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec'
] as const;
/** Upper bound of any monthly total, for the fixed-point unit of the accumulators. */
export const MAXIMUM_MONTHLY_PRECIPITATION = 2000;

export type ClimateStack = {
  columns: number;
  rows: number;
  cellCount: number;
  /** Interpolated station rainfall of 2024-09-27 in millimeters, per cell. */
  helene: Float32Array;
  /** Wetness in [0, 1] per cell. */
  wetness: Float32Array;
  /** Monthly precipitation in millimeters, `cell * MONTH_COUNT + month`, month 0 is January of year 0. */
  precipitation: Float32Array;
};

/** Standard normal by Box-Muller. */
function getNormal(random: Random): number {
  return Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12))) * Math.cos(2 * Math.PI * random());
}

/** Gamma(shape >= 1, scale 1) by Marsaglia and Tsang. */
function getGamma(random: Random, shape: number): number {
  const offset = shape - 1 / 3;
  const factor = 1 / Math.sqrt(9 * offset);
  for (;;) {
    const normal = getNormal(random);
    const cube = (1 + factor * normal) ** 3;
    if (cube <= 0) continue;
    if (
      Math.log(Math.max(random(), 1e-12)) <
      0.5 * normal * normal + offset - offset * cube + offset * Math.log(cube)
    ) {
      return offset * cube;
    }
  }
}

/** Inverse-distance interpolation of the station rainfall onto the cell centers of the world. */
function interpolateRain(
  random: Random,
  world: World,
  columns: number,
  rows: number
): Float32Array {
  const stations = getRainStations(random, world, 600, true);
  const helene = new Float32Array(columns * rows);
  const neighbors = 6;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const x = ((column + 0.5) / columns) * world.width;
      const y = (1 - (row + 0.5) / rows) * world.height;
      const nearest: {distance: number; rain: number}[] = [];
      for (let station = 0; station < stations.count; station++) {
        const distance = Math.hypot(
          stations.positions[station * 2] - x,
          stations.positions[station * 2 + 1] - y
        );
        nearest.push({distance, rain: stations.rain[station]});
      }
      nearest.sort((left, right) => left.distance - right.distance);
      let weightSum = 0;
      let total = 0;
      for (const {distance, rain} of nearest.slice(0, neighbors)) {
        const weight = 1 / (distance * distance + 1e-4);
        weightSum += weight;
        total += weight * rain;
      }
      helene[row * columns + column] = total / weightSum;
    }
  }
  return helene;
}

export function createClimateStack(random: Random, world: World, rows = 16): ClimateStack {
  const columns = Math.max(8, Math.round((rows * world.width) / world.height));
  const cellCount = columns * rows;
  const helene = interpolateRain(random, world, columns, rows);
  const wetnessField = createSmoothField(random, 3, 2);
  const shapeField = createSmoothField(random, 3, 2);
  const monthFields = Array.from({length: MONTH_COUNT}, () => createSmoothField(random, 3, 3));
  const wetness = new Float32Array(cellCount);
  const precipitation = new Float32Array(cellCount * MONTH_COUNT);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const cell = row * columns + column;
      wetness[cell] = wetnessField((column + 0.5) / columns, (row + 0.5) / rows);
      const shape = 20 + 12 * shapeField((column + 0.5) / columns, (row + 0.5) / rows);
      for (let month = 0; month < MONTH_COUNT; month++) {
        const season = 1 + 0.55 * Math.cos((2 * Math.PI * ((month % 12) - HELENE_MONTH)) / 12);
        const mean = (45 + 70 * wetness[cell]) * season;
        // The weather of a month is coherent in space: a smooth factor scales the gamma draw.
        const coherent = Math.exp(
          4.5 * (monthFields[month]((column + 0.5) / columns, (row + 0.5) / rows) - 0.5)
        );
        precipitation[cell * MONTH_COUNT + month] =
          (mean / shape) * getGamma(random, shape) * coherent;
      }
      // Hurricane Helene: the observed footprint lands on the final September.
      precipitation[cell * MONTH_COUNT + (YEAR_COUNT - 1) * 12 + HELENE_MONTH] +=
        1.4 * helene[cell];
    }
  }
  return {columns, rows, cellCount, helene, wetness, precipitation};
}

/** Cell-major copy of months `start` to `start + length` of the record. */
export function getMonthChunk(stack: ClimateStack, start: number, length: number): Float32Array {
  const chunk = new Float32Array(stack.cellCount * length);
  for (let cell = 0; cell < stack.cellCount; cell++) {
    for (let slice = 0; slice < length; slice++) {
      chunk[cell * length + slice] = stack.precipitation[cell * MONTH_COUNT + start + slice];
    }
  }
  return chunk;
}

/** Calendar month (the slot) of each month from `start`. */
export function getMonthSlots(start: number, length: number): Uint32Array {
  return Uint32Array.from({length}, (_, index) => (start + index) % 12);
}

/**
 * A synthetic daily precipitation year in millimeters, `cell * 366 + day`. A smooth weather field
 * that persists from day to day (a blend of yesterday's and a fresh field) decides where it rains:
 * a cell is wet when the field exceeds a threshold that falls with the cell's wetness and the
 * interpolated Helene rainfall, and wet-day amounts are exponential, scaled by the field.
 */
export function createDailyYear(random: Random, stack: ClimateStack): Float32Array {
  const days = 366;
  const daily = new Float32Array(stack.cellCount * days);
  const maximumHelene = Math.max(...stack.helene, 1);
  const weather = new Float32Array(stack.cellCount).fill(0.5);
  for (let day = 0; day < days; day++) {
    const field = createSmoothField(random, 2, 3);
    for (let cell = 0; cell < stack.cellCount; cell++) {
      const column = cell % stack.columns;
      const row = Math.floor(cell / stack.columns);
      weather[cell] =
        0.7 * weather[cell] + 0.3 * field((column + 0.5) / stack.columns, (row + 0.5) / stack.rows);
      const wetShare = 0.2 + 0.3 * stack.wetness[cell] + 0.2 * (stack.helene[cell] / maximumHelene);
      const wet = weather[cell] + 0.08 * (random() - 0.5) > 0.82 - 0.75 * wetShare;
      daily[cell * days + day] = wet
        ? -Math.log(Math.max(random(), 1e-9)) * (4 + 14 * wetShare) * (0.5 + weather[cell])
        : 0;
    }
  }
  return daily;
}
