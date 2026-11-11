// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Cell-major time stacks for the raster time-series runs. The canvas is split: the left part holds
// the probed pixel's charts and the right part the map. The map grid spans the whole drawing world
// (a raster layer fills it); its chart columns hold no data and are clipped away.
//
// Two stacks:
//  - the measured Dixie NDVI stack (16 Sentinel-2 scenes, summer only), and
//  - a synthetic seasonal stack whose summer level and fire ratio are measured NDVI. It adds a
//    seasonal cycle, cloud dips and gaps from a seeded generator, so annual-cycle operations have a
//    full year to fit. Captions of the runs that use it say so.

import {sampleNdviSeries} from '../../../../data/ndvi';
import type {Random} from '../../../inputs';
import {
  DIXIE_FIRST_POST_FIRE_SCENE,
  DIXIE_SCENE_DATES,
  getAxisYears,
  getSceneSeconds,
  SECONDS_PER_DAY
} from './axis';

/** Share of the world width, from the left, given to the charts. */
export const CHART_FRACTION = 0.44;
export const GRID_ROWS = 20;
const DAYS_PER_YEAR = 365.2425;

export type MapGrid = {
  /** Columns of the whole drawing world, chart columns included. */
  columns: number;
  rows: number;
  chartColumns: number;
  mapColumns: number;
  cellCount: number;
  /** World x where the map starts. */
  mapLeft: number;
};

export function createMapGrid(aspect: number, rows = GRID_ROWS): MapGrid {
  const columns = Math.max(12, Math.round(rows * aspect));
  const chartColumns = Math.round(columns * CHART_FRACTION);
  return {
    columns,
    rows,
    chartColumns,
    mapColumns: columns - chartColumns,
    cellCount: columns * rows,
    mapLeft: (aspect * chartColumns) / columns
  };
}

/** Index of the cell's column within the map, or a negative number inside the chart area. */
export function getMapColumn(grid: MapGrid, cell: number): number {
  return (cell % grid.columns) - grid.chartColumns;
}

/** Copies map-sized cell-major data into the whole grid; chart columns are NaN. */
export function placeOnGrid(
  grid: MapGrid,
  mapValues: Float32Array,
  entryCount: number
): Float32Array {
  const output = new Float32Array(grid.cellCount * entryCount).fill(Number.NaN);
  for (let row = 0; row < grid.rows; row++) {
    for (let column = 0; column < grid.mapColumns; column++) {
      const source = (row * grid.mapColumns + column) * entryCount;
      const target = (row * grid.columns + grid.chartColumns + column) * entryCount;
      output.set(mapValues.subarray(source, source + entryCount), target);
    }
  }
  return output;
}

export type DixieStack = {
  grid: MapGrid;
  sliceCount: number;
  /** Seconds of each scene. */
  times: BigInt64Array;
  years: Float32Array;
  /** Cell-major NDVI, NaN for cloud and for chart columns. */
  observed: Float32Array;
};

/** The measured Dixie stack on the map grid. */
export function createDixieStack(aspect: number): DixieStack {
  const grid = createMapGrid(aspect);
  const sliceCount = DIXIE_SCENE_DATES.length;
  const dates = Array.from({length: sliceCount}, (_, index) => index);
  const measured = sampleNdviSeries(grid.mapColumns, grid.rows, dates);
  const times = getSceneSeconds(DIXIE_SCENE_DATES);
  return {
    grid,
    sliceCount,
    times,
    years: getAxisYears(times),
    observed: placeOnGrid(grid, measured, sliceCount)
  };
}

/** One slice of a cell-major stack as a map. */
export function getSliceMap(
  stack: {grid: MapGrid; sliceCount: number; observed: Float32Array},
  slice: number
): Float32Array {
  return Float32Array.from(
    {length: stack.grid.cellCount},
    (_, cell) => stack.observed[cell * stack.sliceCount + slice]
  );
}

export type SeasonalOptions = {
  sliceCount: number;
  yearCount: number;
  /** Date jitter as a fraction of the nominal step, at most 0.35; 0 is regular. */
  jitter?: number;
  /** Seasonal peak shift from the left to the right edge of the map, in years. */
  phaseDrift?: number;
  /** Map columns (fraction of the map width) from which two seasons per year are generated. */
  doubleSeasonFrom?: number;
  /** First slice after a disturbance; the level then falls by the measured fire ratio. */
  breakSlice?: number;
};

export type SeasonalStack = {
  grid: MapGrid;
  sliceCount: number;
  times: BigInt64Array;
  years: Float32Array;
  /** Fraction of the year at each slice, 0 at 1 January. */
  yearFractions: Float32Array;
  /** Noise-free value, cell-major; NaN in chart columns. */
  clean: Float32Array;
  /** Summer NDVI of each cell. */
  level: Float32Array;
  /** Cell-major `(cloud draw, cloud depth, noise)` per observation. */
  draws: Float32Array;
  /** Cloudy and missing observations for the given fractions, cell-major. */
  observe(cloudFraction: number, gapFraction: number): Float32Array;
};

function getMean(
  values: ArrayLike<number>,
  indices: readonly number[],
  stride: number,
  cell: number
) {
  let total = 0;
  let count = 0;
  for (const index of indices) {
    const value = values[cell * stride + index];
    if (Number.isFinite(value)) {
      total += value;
      count++;
    }
  }
  return count > 0 ? total / count : Number.NaN;
}

export function createSeasonalStack(
  aspect: number,
  random: Random,
  options: SeasonalOptions
): SeasonalStack {
  const grid = createMapGrid(aspect);
  const {sliceCount, yearCount} = options;
  const jitter = Math.min(options.jitter ?? 0.3, 0.35);
  const measured = sampleNdviSeries(
    grid.mapColumns,
    grid.rows,
    Array.from({length: DIXIE_SCENE_DATES.length}, (_, index) => index)
  );
  const sceneCount = DIXIE_SCENE_DATES.length;
  const mapCells = grid.mapColumns * grid.rows;
  const before = [0, 1, 2, 3, 4, 5, 6, 7];
  const after = [DIXIE_FIRST_POST_FIRE_SCENE, 9];
  const mapLevel = new Float32Array(mapCells);
  const mapRatio = new Float32Array(mapCells);
  for (let cell = 0; cell < mapCells; cell++) {
    const level = getMean(measured, before, sceneCount, cell);
    const post = getMean(measured, after, sceneCount, cell);
    mapLevel[cell] = Number.isFinite(level) ? Math.min(0.85, Math.max(0.25, level)) : 0.5;
    mapRatio[cell] = Number.isFinite(post / level)
      ? Math.min(1.1, Math.max(0.25, post / level))
      : 1;
  }
  const level = placeOnGrid(grid, mapLevel, 1);
  const ratio = placeOnGrid(grid, mapRatio, 1);

  const stepSeconds = (yearCount * DAYS_PER_YEAR * SECONDS_PER_DAY) / sliceCount;
  const startSeconds = Date.parse('2019-01-01T00:00:00Z') / 1000;
  const offsets = Array.from(
    {length: sliceCount},
    (_, slice) =>
      slice + (slice === 0 || slice === sliceCount - 1 ? 0 : (random() * 2 - 1) * jitter)
  );
  const times = BigInt64Array.from(offsets, offset =>
    BigInt(Math.round(startSeconds + offset * stepSeconds))
  );
  const yearFractions = Float32Array.from(offsets, offset => {
    const date = new Date((startSeconds + offset * stepSeconds) * 1000);
    const yearStart = Date.UTC(date.getUTCFullYear(), 0, 1);
    return (date.getTime() - yearStart) / (DAYS_PER_YEAR * SECONDS_PER_DAY * 1000);
  });

  const clean = new Float32Array(grid.cellCount * sliceCount).fill(Number.NaN);
  const draws = new Float32Array(grid.cellCount * sliceCount * 3);
  for (let draw = 0; draw < draws.length; draw++) draws[draw] = random();
  const doubleFrom = options.doubleSeasonFrom ?? 2;
  for (let row = 0; row < grid.rows; row++) {
    for (let mapColumn = 0; mapColumn < grid.mapColumns; mapColumn++) {
      const cell = row * grid.columns + grid.chartColumns + mapColumn;
      const summer = level[cell];
      const winter = 0.45 * summer;
      const drift = (options.phaseDrift ?? 0) * (mapColumn / Math.max(1, grid.mapColumns - 1));
      const cycles = mapColumn / grid.mapColumns >= doubleFrom ? 2 : 1;
      for (let slice = 0; slice < sliceCount; slice++) {
        const season =
          0.5 + 0.5 * Math.cos(2 * Math.PI * cycles * (yearFractions[slice] - 0.55 - drift));
        const disturbed = options.breakSlice !== undefined && slice >= options.breakSlice;
        clean[cell * sliceCount + slice] =
          (winter + (summer - winter) * season) * (disturbed ? ratio[cell] : 1);
      }
    }
  }

  return {
    grid,
    sliceCount,
    times,
    years: getAxisYears(times),
    yearFractions,
    clean,
    level,
    draws,
    observe(cloudFraction, gapFraction) {
      const observed = new Float32Array(clean.length);
      for (let index = 0; index < clean.length; index++) {
        const cloud = draws[index * 3];
        const noise = (draws[index * 3 + 2] - 0.5) * 0.04;
        let value = clean[index] + noise;
        if (cloud < cloudFraction) value -= 0.1 + 0.3 * draws[index * 3 + 1] * Math.max(value, 0.2);
        if (cloud > 1 - gapFraction) value = Number.NaN;
        observed[index] = Number.isFinite(clean[index]) ? Math.max(value, 0.02) : Number.NaN;
      }
      return observed;
    }
  };
}
