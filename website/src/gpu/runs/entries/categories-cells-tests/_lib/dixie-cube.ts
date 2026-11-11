// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The Dixie fire NDVI stack (Sentinel-2, 16 real scenes from 2017-09-02 to 2024-08-01) as a space-time
// cube for the change point runs: 48 x 30 cells, cell-major. The fire burned between the scenes of
// 2021-07-13 and 2021-09-11 (slice 8). The display grid is wider than the data so the map sits on the
// left and a chart fits on the right. Scene dates are irregular and are drawn at their true spacing.

import {sampleNdviSeries} from '../../../../data/ndvi';

export const CUBE_COLUMNS = 48;
export const CUBE_ROWS = 30;
export const CUBE_CELL_COUNT = CUBE_COLUMNS * CUBE_ROWS;
export const DISPLAY_COLUMNS = 80;
export const SCENE_DATES = [
  '2017-09-02',
  '2018-07-29',
  '2018-08-08',
  '2019-08-23',
  '2020-06-03',
  '2020-08-02',
  '2021-06-03',
  '2021-07-13',
  '2021-09-11',
  '2021-09-21',
  '2022-06-23',
  '2022-08-07',
  '2023-07-13',
  '2023-08-02',
  '2024-06-22',
  '2024-08-01'
] as const;
export const SCENE_DAYS = SCENE_DATES.map(date => Date.parse(date) / 86400000);
/** Plot rectangle of the series chart on the right (fractions of the view, y up). */
export const SERIES_RECTANGLE = [0.66, 0.16, 0.97, 0.84] as const;
export const NDVI_RANGE = [-0.2, 1] as const;

/** Cell-major NDVI of the first `sliceCount` scenes; NaN where a scene has no valid pixel. */
export function createDixieCube(sliceCount: number): Float32Array {
  return sampleNdviSeries(
    CUBE_COLUMNS,
    CUBE_ROWS,
    Array.from({length: sliceCount}, (_, scene) => scene)
  );
}

/** Lays a per-cell value on the wider display grid, with NaN outside the data. */
export function padToDisplay(values: ArrayLike<number>): Float32Array {
  const display = new Float32Array(DISPLAY_COLUMNS * CUBE_ROWS).fill(Number.NaN);
  for (let row = 0; row < CUBE_ROWS; row++) {
    for (let column = 0; column < CUBE_COLUMNS; column++) {
      display[row * DISPLAY_COLUMNS + column] = values[row * CUBE_COLUMNS + column];
    }
  }
  return display;
}

/** The data cell under a world position, or -1 outside the data. */
export function getDataCell(x: number, y: number, world: {width: number; height: number}): number {
  const column = Math.floor((x / world.width) * DISPLAY_COLUMNS);
  const row = Math.floor((1 - y / world.height) * CUBE_ROWS);
  if (column < 0 || column >= CUBE_COLUMNS || row < 0 || row >= CUBE_ROWS) return -1;
  return row * CUBE_COLUMNS + column;
}

/** World rectangle of a data cell. */
export function getCellBox(
  cell: number,
  world: {width: number; height: number}
): {minimum: [number, number]; maximum: [number, number]} {
  const column = cell % CUBE_COLUMNS;
  const row = Math.floor(cell / CUBE_COLUMNS);
  const width = world.width / DISPLAY_COLUMNS;
  const height = world.height / CUBE_ROWS;
  return {
    minimum: [column * width, world.height - (row + 1) * height],
    maximum: [(column + 1) * width, world.height - row * height]
  };
}

/** World x of a day on the series chart of `sliceCount` scenes (true spacing). */
export function getSeriesX(day: number, sliceCount: number, world: {width: number}): number {
  const first = SCENE_DAYS[0];
  const last = SCENE_DAYS[sliceCount - 1];
  const fraction = (day - first) / (last - first);
  const [left, , right] = SERIES_RECTANGLE;
  return (left + fraction * (right - left)) * world.width;
}

/** World y of an NDVI value on the series chart. */
export function getSeriesY(value: number, world: {height: number}): number {
  const [, bottom, , top] = SERIES_RECTANGLE;
  const fraction = (value - NDVI_RANGE[0]) / (NDVI_RANGE[1] - NDVI_RANGE[0]);
  return (bottom + fraction * (top - bottom)) * world.height;
}

/** Position of each scene along the chart's x axis, 0 to 1, at true date spacing. */
export function getSceneFractions(sliceCount: number): Float32Array {
  const first = SCENE_DAYS[0];
  const last = SCENE_DAYS[sliceCount - 1];
  return Float32Array.from(
    {length: sliceCount},
    (_, scene) => (SCENE_DAYS[scene] - first) / (last - first)
  );
}
