// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The sixteen Sentinel-2 acquisitions over the Dixie Fire area (2017-09-02 to 2024-08-01) as a
// cell-major NDVI stack, with the day of year and year of every scene and the distance in cells
// from each observation to the nearest cloud, shadow or no-data cell of its scene (the QA mask).

import {sampleNdviSeries} from '../../../../data/ndvi';

export const ACQUISITION_DATES = [
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

const MILLISECONDS_PER_DAY = 86_400_000;

/** `[dayOfYear, year]` per scene, the layout of `sliceAttributes`. */
export function getSliceAttributes(): Float32Array {
  const attributes = new Float32Array(ACQUISITION_DATES.length * 2);
  ACQUISITION_DATES.forEach((date, slice) => {
    const time = Date.parse(`${date}T00:00:00Z`);
    const year = new Date(time).getUTCFullYear();
    attributes[slice * 2] = Math.round((time - Date.UTC(year, 0, 1)) / MILLISECONDS_PER_DAY) + 1;
    attributes[slice * 2 + 1] = year;
  });
  return attributes;
}

export type AcquisitionStack = {
  columns: number;
  rows: number;
  cellCount: number;
  sliceCount: number;
  /** NDVI, `cell * sliceCount + slice`; NaN where the QA mask rejected the observation. */
  observations: Float32Array;
  /** QA flag per observation, 1 where the observation is usable. */
  validity: Uint32Array;
  /** Cells to the nearest rejected cell of the same scene; Infinity when the scene has none. */
  cloudDistance: Float32Array;
};

export function createAcquisitionStack(aspect: number, rows: number): AcquisitionStack {
  const columns = Math.max(8, Math.round(rows * aspect));
  const cellCount = columns * rows;
  const sliceCount = ACQUISITION_DATES.length;
  const dateIndices = Array.from({length: sliceCount}, (_, index) => index);
  const observations = sampleNdviSeries(columns, rows, dateIndices);
  const validity = Uint32Array.from(observations, value => (Number.isFinite(value) ? 1 : 0));
  const cloudDistance = new Float32Array(cellCount * sliceCount).fill(Number.POSITIVE_INFINITY);
  for (let slice = 0; slice < sliceCount; slice++) {
    const rejected: number[] = [];
    for (let cell = 0; cell < cellCount; cell++) {
      if (!validity[cell * sliceCount + slice]) rejected.push(cell);
    }
    for (let cell = 0; cell < cellCount && rejected.length > 0; cell++) {
      const row = Math.floor(cell / columns);
      const column = cell % columns;
      let nearest = Number.POSITIVE_INFINITY;
      for (const other of rejected) {
        nearest = Math.min(
          nearest,
          Math.hypot(row - Math.floor(other / columns), column - (other % columns))
        );
      }
      cloudDistance[cell * sliceCount + slice] = nearest;
    }
  }
  return {columns, rows, cellCount, sliceCount, observations, validity, cloudDistance};
}
